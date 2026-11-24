import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as wasm from '@gavmor/sipkit-wasm';
import { SipModelRunner, parseSipModel } from '../src/index.js';

wasm.initSync({ module: readFileSync(new URL('../../wasm/pkg/sipkit_wasm_bg.wasm', import.meta.url)) });

const hdrRng = (name: string, varId: number) => ({
  name,
  function: 'HDR_2_0',
  arguments: { counter: 'PM_Index', entity: 1, varId, seed3: 3, seed4: 4 },
});

const metalogSip = (name: string, ref: object, aCoefficients: number[]) => ({
  name,
  ref,
  function: 'Metalog_1_0',
  arguments: { aCoefficients },
});

describe('parseSipModel', () => {
  it('accepts a minimal model and rejects an empty sips array', () => {
    const model = {
      name: 'm',
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [metalogSip('x', { source: 'rng', name: 'r1' }, [0, 1])],
    };
    expect(parseSipModel(model).name).toBe('m');
    expect(() => parseSipModel({ name: 'm', U01: {}, sips: [] })).toThrow();
  });

  it('strict mode requires objectType "sipModel"', () => {
    const model = {
      name: 'm',
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [metalogSip('x', { source: 'rng', name: 'r1' }, [0, 1])],
    };
    expect(() => parseSipModel(model, { strict: true })).toThrow(/objectType/);
    expect(parseSipModel({ ...model, objectType: 'sipModel' }, { strict: true }).name).toBe('m');
  });
});

describe('SipModelRunner', () => {
  it('evaluates an HDR-driven Metalog through the fused WASM path', async () => {
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [metalogSip('x', { source: 'rng', name: 'r1' }, [0, 1])],
    });
    const got = await runner.evaluateSip('x', { count: 100 });
    const want = wasm.hdrMetalogBatch(1, 100, 1, 2, 3, 4, Float64Array.of(0, 1), undefined, undefined, 0);
    expect([...got]).toEqual([...want]);
  });

  it('resolves parameter-ref seeds at runtime', async () => {
    const rng = {
      name: 'r1',
      function: 'HDR_2_0',
      arguments: {
        counter: 'PM_Index',
        entity: { type: 'parameter', name: 'ent' },
        varId: 2,
        seed3: 3,
        seed4: 4,
      },
    };
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      U01: { rng: [rng] },
      sips: [metalogSip('x', { source: 'rng', name: 'r1' }, [0, 1])],
    });
    await expect(runner.evaluateSip('x', { count: 10 })).rejects.toThrow(/Missing required runtime parameter "ent"/);
    const got = await runner.evaluateSip('x', { count: 10, parameters: { ent: 1 } });
    const want = wasm.hdrMetalogBatch(1, 10, 1, 2, 3, 4, Float64Array.of(0, 1), undefined, undefined, 0);
    expect([...got]).toEqual([...want]);
  });

  it('evaluates a Gaussian copula from a sparse upper-triangular matrix', async () => {
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      globalVariables: [
        {
          name: 'R',
          value: {
            columns: ['a', 'b'],
            rows: ['a', 'b'],
            matrix: [{ row: 'a', col: 'b', value: 0.5 }],
          },
        },
      ],
      U01: {
        rng: [hdrRng('r1', 2), hdrRng('r2', 3)],
        copula: [
          {
            name: 'c',
            function: 'GaussianCopula',
            arguments: { correlationMatrix: { type: 'globalVariables', value: 'R' }, rng: ['r1', 'r2'] },
            copulaLayer: ['a', 'b'],
          },
        ],
      },
      sips: [
        metalogSip('x', { source: 'copula', name: 'c', copulaLayer: 'a' }, [0, 1]),
        metalogSip('y', { source: 'copula', name: 'c', copulaLayer: 'b' }, [10, 2]),
      ],
    });
    const count = 1000;
    const results = await runner.evaluateAll({ count });
    const L = wasm.choleskyDecompose(2, Float64Array.of(1, 0.5, 0.5, 1));
    const layers = wasm.gaussianCopulaBatch(
      2,
      1,
      count,
      Uint32Array.of(1, 2, 3, 4, 1, 3, 3, 4),
      L,
    );
    expect([...results['x']!]).toEqual([
      ...wasm.metalogBatch(layers.subarray(0, count), Float64Array.of(0, 1), undefined, undefined, 0),
    ]);
    expect([...results['y']!]).toEqual([
      ...wasm.metalogBatch(
        layers.subarray(count, 2 * count),
        Float64Array.of(10, 2),
        undefined,
        undefined,
        0,
      ),
    ]);
  });

  it('resolves GeneralizedMetalog rows tolerating the numberofIIDs casing quirk', async () => {
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      globalVariables: [
        {
          name: 'T',
          value: [
            { aCoefficients: [0, 1], numberofIIDs: 3 },
            { aCoefficients: [5, 2], numberofIIDs: 5 },
          ],
        },
      ],
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [
        {
          name: 'g',
          ref: { source: 'rng', name: 'r1' },
          function: 'GeneralizedMetalog',
          arguments: {
            aCoefficients: { type: 'globalVariables', value: 'T' },
            dims: [{ type: 'parameter', name: 'numberOfIIDs', numericType: 'integer' }],
          },
        },
      ],
    });
    const got = await runner.evaluateSip('g', { count: 10, parameters: { numberOfIIDs: 3 } });
    const want = wasm.hdrMetalogBatch(1, 10, 1, 2, 3, 4, Float64Array.of(0, 1), undefined, undefined, 0);
    expect([...got]).toEqual([...want]);
    await expect(
      runner.evaluateSip('g', { count: 10, parameters: { numberOfIIDs: 4 } }),
    ).rejects.toThrow(/No matching GeneralizedMetalog coefficient row/);
    await expect(runner.evaluateSip('g', { count: 10 })).rejects.toThrow(
      /Missing required GeneralizedMetalog parameter "numberOfIIDs"/,
    );
  });

  it('evaluates a Lookup_Table through the fused HDR path', async () => {
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [
        {
          name: 'lk',
          ref: { source: 'rng', name: 'r1' },
          function: 'Lookup_Table',
          arguments: {
            value: [
              [0.25, 10],
              [0.5, 20],
              [1.0, 30],
            ],
          },
        },
      ],
    });
    const got = await runner.evaluateSip('lk', { count: 100 });
    const want = wasm.hdrLookupBatch(
      1,
      100,
      1,
      2,
      3,
      4,
      Float64Array.of(0.25, 0.5, 1.0),
      Float64Array.of(10, 20, 30),
    );
    expect([...got]).toEqual([...want]);
  });

  it('slices an inline SIP_Array by PM_Index', async () => {
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [
        {
          name: 'arr',
          ref: { source: 'rng', name: 'r1' },
          function: 'SIP_Array',
          arguments: { type: 'array', value: [10, 11, 12, 13, 14, 15, 16, 17] },
        },
      ],
    });
    expect([...(await runner.evaluateSip('arr', { count: 3, startCounter: 3 }))]).toEqual([
      12, 13, 14,
    ]);
  });

  it('prefers the portable SIP over an environment-specific override', async () => {
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [
        {
          name: 'x',
          environment: 'Excel',
          ref: { source: 'rng', name: 'r1' },
          function: 'Metalog_1_0',
          arguments: { aCoefficients: [5, 1] },
        },
        metalogSip('x', { source: 'rng', name: 'r1' }, [0, 1]),
      ],
    });
    expect(runner.sipNames).toEqual(['x']);
    const got = await runner.evaluateSip('x', { count: 10 });
    const want = wasm.hdrMetalogBatch(1, 10, 1, 2, 3, 4, Float64Array.of(0, 1), undefined, undefined, 0);
    expect([...got]).toEqual([...want]);
  });

  it('summarizes an HDR-driven Metalog without allocating trials', async () => {
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [metalogSip('x', { source: 'rng', name: 'r1' }, [0, 1])],
    });
    const stats = await runner.summarizeSip('x', { count: 10_000 });
    const trials = await runner.evaluateSip('x', { count: 10_000 });
    const mean = trials.reduce((s, v) => s + v, 0) / trials.length;
    expect(stats.count).toBe(10_000);
    expect(stats.nanCount).toBe(0);
    expect(Math.abs(stats.mean - mean)).toBeLessThan(1e-9);
  });

  it('rejects unknown RNG references', async () => {
    const runner = new SipModelRunner(wasm, {
      name: 'm',
      U01: { rng: [hdrRng('r1', 2)] },
      sips: [metalogSip('x', { source: 'rng', name: 'nope' }, [0, 1])],
    });
    await expect(runner.evaluateSip('x', { count: 10 })).rejects.toThrow(
      /Unknown RNG reference "nope"/,
    );
  });
});
