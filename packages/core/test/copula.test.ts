import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  cholesky,
  correlatedUniforms,
  GaussianCopula,
  Metalog,
  normalCdf,
  normalQuantile,
  type HdrSeeds,
  type MetalogBounds,
} from '../src/index.js';

function matMul(a: number[][], b: number[][]): number[][] {
  return a.map((row) =>
    b[0]!.map((_, j) => row.reduce((s, v, k) => s + v * b[k]![j]!, 0)),
  );
}
function transpose(a: number[][]): number[][] {
  return a[0]!.map((_, j) => a.map((row) => row[j]!));
}
function close(a: number[][], b: number[][], tol: number): boolean {
  return a.every((row, i) => row.every((v, j) => Math.abs(v - b[i]![j]!) <= tol));
}
function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i]! - mx) * (ys[i]! - my);
    sxx += (xs[i]! - mx) ** 2;
    syy += (ys[i]! - my) ** 2;
  }
  return sxy / Math.sqrt(sxx * syy);
}

describe('normalQuantile / normalCdf', () => {
  it('inverts the CDF at known points', () => {
    expect(normalQuantile(0.5)).toBeCloseTo(0, 12);
    expect(normalQuantile(0.975)).toBeCloseTo(1.959963984540054, 8);
    expect(normalQuantile(0.025)).toBeCloseTo(-1.959963984540054, 8);
    expect(normalQuantile(0.001)).toBeCloseTo(-3.090232306167813, 8);
    expect(normalQuantile(0.999)).toBeCloseTo(3.090232306167813, 8);
  });

  it('returns NaN outside (0, 1)', () => {
    for (const p of [0, 1, -0.1, 1.1, NaN]) expect(normalQuantile(p)).toBeNaN();
  });

  it('round-trips through the CDF', () => {
    for (const z of [-2.5, -1, -0.25, 0, 0.25, 1, 2.5]) {
      // Limited by normalCdf's A&S 7.1.26 accuracy (|error| < 1.5e-7).
      expect(normalQuantile(normalCdf(z))).toBeCloseTo(z, 5);
    }
  });
});

describe('cholesky', () => {
  it('factors known matrices', () => {
    expect(cholesky([[4, 2], [2, 3]])).toEqual([
      [2, 0],
      [1, Math.SQRT2],
    ]);
    const L = cholesky([[1, 0.5], [0.5, 1]]);
    expect(L[0]).toEqual([1, 0]);
    expect(L[1]![0]).toBeCloseTo(0.5, 12);
    expect(L[1]![1]).toBeCloseTo(Math.sqrt(0.75), 12);
  });

  it('reconstructs R = L Lᵀ', () => {
    const R = [
      [1, 0.8, 0.3],
      [0.8, 1, 0.5],
      [0.3, 0.5, 1],
    ];
    const L = cholesky(R);
    expect(close(matMul(L, transpose(L)), R, 1e-12)).toBe(true);
  });

  it('accepts the singular ρ = 1 matrix', () => {
    const L = cholesky([[1, 1], [1, 1]]);
    expect(close(matMul(L, transpose(L)), [[1, 1], [1, 1]], 1e-12)).toBe(true);
  });

  it('rejects non-symmetric, non-PSD, and non-square input', () => {
    expect(() => cholesky([[1, 0.5], [0.6, 1]])).toThrow(RangeError);
    expect(() => cholesky([[1, 2], [2, 1]])).toThrow(RangeError);
    expect(() => cholesky([[1, 0.5]])).toThrow(RangeError);
    expect(() => cholesky([])).toThrow(RangeError);
  });
});

describe('GaussianCopula', () => {
  const seeds = { entity: 1, varId: 2, seed3: 3, seed4: 4 };

  it('is reproducible from (counter, seeds) and varies by counter', () => {
    const cop = new GaussianCopula(
      [[1, 0.5], [0.5, 1]],
      [new Metalog([0, 1]), new Metalog([10, 2])],
    );
    expect(cop.sample(7, seeds)).toEqual(cop.sample(7, seeds));
    expect(cop.sample(7, seeds)).not.toEqual(cop.sample(8, seeds));
  });

  it('recovers the target correlation on the normal scores', () => {
    const rho = 0.8;
    const cop = new GaussianCopula(
      [[1, rho], [rho, 1]],
      [new Metalog([0, 1]), new Metalog([0, 1])],
    );
    const n = 20000;
    const us = Array.from({ length: n }, (_, i) => cop.sampleUniforms(1 + i, seeds));
    // Correlated uniforms -> normal scores; their Pearson r should be ~rho.
    const zs = us.map(([u, v]) => [normalQuantile(u!), normalQuantile(v!)] as const);
    const r = pearson(
      zs.map(([z]) => z),
      zs.map(([, z]) => z),
    );
    expect(Math.abs(r - rho)).toBeLessThan(0.03);
  });

  it('preserves the marginals', () => {
    const cop = new GaussianCopula(
      [[1, 0.7], [0.7, 1]],
      [
        new Metalog([0, 1]),
        // Lower-bounded Metalog works in log space: a1 = ln(median - lower).
        new Metalog([Math.log(100), 0.2], { kind: 'lower', lower: 0 }),
      ],
    );
    const samples = cop.sampleMany(1, 20000, seeds);
    const xs = samples.map(([x]) => x!).sort((a, b) => a - b);
    const ys = samples.map(([, y]) => y!).sort((a, b) => a - b);
    // Median of a standard-normal Metalog [0, 1] is 0; of the second, ~100.
    expect(Math.abs(xs[Math.floor(xs.length / 2)]!)).toBeLessThan(0.05);
    expect(Math.abs(ys[Math.floor(ys.length / 2)]! - 100)).toBeLessThan(1.5);
  });

  it('reduces to independence at ρ = 0', () => {
    const cop = new GaussianCopula(
      [[1, 0], [0, 1]],
      [new Metalog([0, 1]), new Metalog([0, 1])],
    );
    const us = Array.from({ length: 20000 }, (_, i) => cop.sampleUniforms(1 + i, seeds));
    const r = pearson(
      us.map(([u]) => normalQuantile(u!)),
      us.map(([, v]) => normalQuantile(v!)),
    );
    expect(Math.abs(r)).toBeLessThan(0.03);
  });

  it('rejects a non-unit diagonal and mismatched marginals', () => {
    expect(
      () => new GaussianCopula([[2, 0], [0, 1]], [new Metalog([0, 1]), new Metalog([0, 1])]),
    ).toThrow(RangeError);
    expect(() => new GaussianCopula([[1, 0], [0, 1]], [new Metalog([0, 1])])).toThrow(
      RangeError,
    );
  });

  it('correlatedUniforms validates its inputs', () => {
    const L = cholesky([[1, 0.5], [0.5, 1]]);
    expect(() => correlatedUniforms(L, [0.5])).toThrow(RangeError);
    expect(() => correlatedUniforms(L, [0, 0.5])).toThrow(RangeError);
  });
});

interface CopulaParity {
  cases: {
    correlation: number[][];
    marginals: { a: number[]; bounds: MetalogBounds }[];
    seeds: HdrSeeds;
    factor: number[][];
    samples: { counter: number; uniforms: number[]; x: number[] }[];
  }[];
}

interface NormalAccuracy {
  quantile_tol: number;
  cdf_tol: number;
  quantile: { p: number; z: number }[];
  cdf: { x: number; p: number }[];
}

const parity = JSON.parse(
  readFileSync(join(import.meta.dirname, '../../../fixtures/copula-parity.json'), 'utf8'),
) as CopulaParity;

const accuracy = JSON.parse(
  readFileSync(join(import.meta.dirname, '../../../fixtures/normal-accuracy.json'), 'utf8'),
) as NormalAccuracy;

describe('shared copula parity fixture', () => {
  it('matches every factor, uniform and sample exactly', () => {
    // Exactness is the point: this file is also checked by the Rust and
    // WASM suites, so any transcription drift fails here.
    for (const c of parity.cases) {
      const cop = new GaussianCopula(
        c.correlation,
        c.marginals.map((m) => new Metalog(m.a, m.bounds)),
      );
      expect(cop.factor).toEqual(c.factor);
      for (const s of c.samples) {
        const uniforms = cop.sampleUniforms(s.counter, c.seeds);
        const x = cop.sample(s.counter, c.seeds);
        expect(uniforms).toHaveLength(s.uniforms.length);
        expect(x).toHaveLength(s.x.length);
        s.uniforms.forEach((u, i) => expect(uniforms[i]).toBe(u));
        s.x.forEach((v, i) => expect(x[i]).toBe(v));
      }
    }
  });
});

describe('scipy normal accuracy fixture', () => {
  it('quantile stays within the fixture tolerance', () => {
    for (const { p, z } of accuracy.quantile) {
      expect(Math.abs(normalQuantile(p) - z)).toBeLessThanOrEqual(
        accuracy.quantile_tol * Math.max(1, Math.abs(z)),
      );
    }
  });

  it('CDF stays within the fixture tolerance', () => {
    for (const { x, p } of accuracy.cdf) {
      expect(Math.abs(normalCdf(x) - p)).toBeLessThanOrEqual(accuracy.cdf_tol);
    }
  });
});
