import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { MAX_SEED, hdrFill, hdrUniform, type HdrSeeds } from '../src/index.js';

interface Golden {
  seeds: Required<HdrSeeds>;
  cases: { counter: number; u: number }[];
}

const golden = JSON.parse(
  readFileSync(join(import.meta.dirname, '../../../fixtures/hdr-2.0-seed-standard.json'), 'utf8'),
) as Golden;

describe('HDR 2.0 golden vectors', () => {
  it('hdrUniform matches every golden exactly', () => {
    expect(golden.cases).toHaveLength(10000);
    for (const { counter, u } of golden.cases) expect(hdrUniform(counter, golden.seeds)).toBe(u);
  });

  it('hdrFill matches every golden exactly', () => {
    const first = golden.cases[0]!.counter;
    const out = hdrFill(golden.seeds, first, golden.cases.length);
    golden.cases.forEach(({ counter, u }, i) => {
      expect(counter).toBe(first + i);
      expect(out[i]).toBe(u);
    });
  });
});

describe('HDR domain', () => {
  const seeds: HdrSeeds = { entity: 1, varId: 2 };

  it.each([-1, 1.5, MAX_SEED + 1, NaN])('rejects %s in every input position', (bad) => {
    expect(() => hdrUniform(bad, seeds)).toThrow(RangeError);
    expect(() => hdrUniform(1, { ...seeds, entity: bad })).toThrow(RangeError);
    expect(() => hdrUniform(1, { ...seeds, varId: bad })).toThrow(RangeError);
    expect(() => hdrUniform(1, { ...seeds, seed3: bad })).toThrow(RangeError);
    expect(() => hdrUniform(1, { ...seeds, seed4: bad })).toThrow(RangeError);
    expect(() => hdrFill({ ...seeds, entity: bad }, 1, 1)).toThrow(RangeError);
  });

  it('accepts MAX_SEED in every position and stays inside (0, 1)', () => {
    const u = hdrUniform(MAX_SEED, { entity: MAX_SEED, varId: MAX_SEED, seed3: MAX_SEED, seed4: MAX_SEED });
    expect(u).toBeGreaterThan(0);
    expect(u).toBeLessThan(1);
  });

  it('rejects hdrFill whose last counter exceeds MAX_SEED', () => {
    expect(hdrFill(seeds, MAX_SEED - 1, 2)).toHaveLength(2);
    expect(() => hdrFill(seeds, MAX_SEED - 1, 3)).toThrow(RangeError);
  });

  it('seed3/seed4 default to 0', () => {
    expect(hdrUniform(7, seeds)).toBe(hdrUniform(7, { ...seeds, seed3: 0, seed4: 0 }));
  });
});
