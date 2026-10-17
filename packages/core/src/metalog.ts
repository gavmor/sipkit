/** Metalog quantile function (Keelin 2016). */

export type MetalogBounds =
  | { kind: 'unbounded' }
  | { kind: 'lower'; lower: number }
  | { kind: 'upper'; upper: number }
  | { kind: 'bounded'; lower: number; upper: number };

const MIN_TERMS = 2;
const MAX_TERMS = 16;

/**
 * Invariants: 2 ≤ k ≤ 16, every coefficient finite, every bound finite,
 * and lower < upper when bounded. Coefficients and bounds are frozen copies.
 */
export class Metalog {
  readonly coefficients: readonly number[];
  readonly bounds: MetalogBounds;

  constructor(coefficients: readonly number[], bounds: MetalogBounds = { kind: 'unbounded' }) {
    const k = coefficients.length;
    if (k < MIN_TERMS || k > MAX_TERMS) {
      throw new RangeError(`metalog needs ${MIN_TERMS}..${MAX_TERMS} coefficients, got ${k}`);
    }
    coefficients.forEach((a, i) => {
      if (!Number.isFinite(a)) throw new RangeError(`coefficient ${i + 1} is not finite: ${a}`);
    });
    switch (bounds.kind) {
      case 'unbounded':
        break;
      case 'lower':
        if (!Number.isFinite(bounds.lower)) throw new RangeError(`lower bound is not finite`);
        break;
      case 'upper':
        if (!Number.isFinite(bounds.upper)) throw new RangeError(`upper bound is not finite`);
        break;
      case 'bounded':
        if (!Number.isFinite(bounds.lower) || !Number.isFinite(bounds.upper)) {
          throw new RangeError(`bounds must be finite`);
        }
        if (!(bounds.lower < bounds.upper)) {
          throw new RangeError(`lower bound ${bounds.lower} must be < upper bound ${bounds.upper}`);
        }
        break;
    }
    this.coefficients = Object.freeze([...coefficients]);
    this.bounds = Object.freeze({ ...bounds });
  }

  /** Quantile at probability y; NaN unless 0 < y < 1. */
  quantile(y: number): number {
    if (!(y > 0 && y < 1)) return NaN;
    const a = this.coefficients;
    const L = Math.log(y) - Math.log1p(-y);
    const d = y - 0.5;
    // Terms 1..4: a1 + a2·L + a3·d·L + a4·d.
    let m = a[0]! + a[1]! * L;
    if (a.length > 2) m += a[2]! * d * L;
    if (a.length > 3) m += a[3]! * d;
    // j ≥ 5: odd j → d^((j−1)/2), even j → d^(j/2−1)·L; each pair shares one power of d.
    let p = d;
    for (let j = 4; j < a.length; j += 2) {
      p *= d;
      m += a[j]! * p;
      if (j + 1 < a.length) m += a[j + 1]! * p * L;
    }
    const b = this.bounds;
    switch (b.kind) {
      case 'unbounded':
        return m;
      case 'lower':
        return b.lower + Math.exp(m);
      case 'upper':
        return b.upper - Math.exp(-m);
      case 'bounded':
        return b.lower + (b.upper - b.lower) / (1 + Math.exp(-m));
    }
  }
}
