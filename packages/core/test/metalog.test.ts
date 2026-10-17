import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { Metalog, type MetalogBounds } from '../src/index.js';

interface Parity {
  cases: { a: number[]; bounds: MetalogBounds; points: { y: number; x: number }[] }[];
}

const parity = JSON.parse(
  readFileSync(join(import.meta.dirname, '../../../fixtures/metalog-quantile.json'), 'utf8'),
) as Parity;

describe('metalog parity vectors', () => {
  it.each(parity.cases.map((c, i) => [i, c] as const))('case %i', (_, { a, bounds, points }) => {
    const m = new Metalog(a, bounds);
    for (const { y, x } of points) {
      expect(Math.abs(m.quantile(y) - x)).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(x)));
    }
  });
});

describe('metalog construction', () => {
  it.each<[string, number[], MetalogBounds]>([
    ['k = 1', [1], { kind: 'unbounded' }],
    ['k = 17', Array.from({ length: 17 }, () => 0.1), { kind: 'unbounded' }],
    ['NaN coefficient', [1, NaN], { kind: 'unbounded' }],
    ['infinite coefficient', [1, Infinity], { kind: 'unbounded' }],
    ['lower = upper', [0, 1], { kind: 'bounded', lower: 2, upper: 2 }],
    ['lower > upper', [0, 1], { kind: 'bounded', lower: 3, upper: 2 }],
    ['infinite lower', [0, 1], { kind: 'lower', lower: -Infinity }],
    ['NaN upper', [0, 1], { kind: 'upper', upper: NaN }],
    ['infinite bounded upper', [0, 1], { kind: 'bounded', lower: 0, upper: Infinity }],
  ])('rejects %s', (_, a, bounds) => {
    expect(() => new Metalog(a, bounds)).toThrow(RangeError);
  });

  it('accepts k = 2 and k = 16', () => {
    expect(() => new Metalog([0, 1])).not.toThrow();
    expect(() => new Metalog(Array.from({ length: 16 }, () => 0.01))).not.toThrow();
  });
});

describe('metalog quantile domain', () => {
  const m = new Metalog([10, 2, 0.5, -1, 0.3]);

  it.each([0, 1, -0.1, 1.1, NaN])('returns NaN at y = %s', (y) => {
    expect(m.quantile(y)).toBeNaN();
  });

  it('respects each bound kind', () => {
    const a = [0, 3, 1, -2, 0.5, 0.2];
    const ys = [1e-12, 1e-6, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-6, 1 - 1e-12];
    const lower = new Metalog(a, { kind: 'lower', lower: -5 });
    const upper = new Metalog(a, { kind: 'upper', upper: 5 });
    const bounded = new Metalog(a, { kind: 'bounded', lower: -5, upper: 5 });
    for (const y of ys) {
      expect(lower.quantile(y)).toBeGreaterThanOrEqual(-5);
      expect(upper.quantile(y)).toBeLessThanOrEqual(5);
      const x = bounded.quantile(y);
      expect(x).toBeGreaterThanOrEqual(-5);
      expect(x).toBeLessThanOrEqual(5);
    }
  });
});
