/**
 * Cholesky factorisation and the Gaussian copula, fitting sipkit's
 * SIPmath 3.1 kernel (HDR 2.0 PRNG + Metalog marginals).
 *
 * Recipe (the standard one — cf. NumPy's `random.multivariate_normal`,
 * which `@chaos-labs/multivariate-normal` ports to JS):
 *
 *   1. z_i = Φ⁻¹(u_i) for independent uniforms u_i (here: HDR uniforms,
 *      one distinct `varId` per dimension, same counter/entity/seeds).
 *   2. w = L z, where L is the lower Cholesky factor of the correlation
 *      matrix R (R = L Lᵀ).
 *   3. v_i = Φ(w_i) — correlated uniforms, each still Uniform(0, 1).
 *   4. x_i = F_i⁻¹(v_i) — here F_i⁻¹ is a Metalog quantile, so the
 *      marginals are preserved exactly and only the dependence is
 *      Gaussian.
 *
 * No dependency is added: the Cholesky factor is ~40 lines, the
 * normal quantile/CDF live in `./normal.js`. That keeps the kernel
 * zero-dependency, unlike pulling in `ml-matrix` (Cholesky only) or
 * `mathjs` (no Cholesky in the core build) for this one routine.
 */
import { hdrUniform, type HdrSeeds } from './hdr.js';
import { type Metalog } from './metalog.js';
import { normalCdf, normalQuantile } from './normal.js';

/** Smallest pivot treated as exactly zero (semidefinite case). */
const PIVOT_TOL = 1e-12;

/** Validate a square, finite, symmetric matrix; return its size. */
function checkSquare(matrix: readonly (readonly number[])[]): number {
  const n = matrix.length;
  if (n === 0) throw new RangeError('matrix must be non-empty');
  for (let i = 0; i < n; i++) {
    const row = matrix[i]!;
    if (row.length !== n) {
      throw new RangeError(`matrix must be square: row ${i} has length ${row.length}, expected ${n}`);
    }
    for (let j = 0; j < n; j++) {
      if (!Number.isFinite(row[j]!)) {
        throw new RangeError(`matrix entry [${i}][${j}] must be finite, got ${row[j]}`);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(matrix[i]![j]! - matrix[j]![i]!) > 1e-10) {
        throw new RangeError(
          `matrix must be symmetric: [${i}][${j}]=${matrix[i]![j]} != [${j}][${i}]=${matrix[j]![i]}`,
        );
      }
    }
  }
  return n;
}

/**
 * Lower Cholesky factor L of a symmetric positive-semidefinite matrix
 * A, so L Lᵀ = A (within floating-point error).
 *
 * Throws RangeError if A is not symmetric positive-semidefinite.
 * Singular (semidefinite) matrices are accepted: a zero pivot yields a
 * zero row/column in L, which is the correct factor (e.g. ρ = 1 gives
 * z₂ = z₁). This is the eigen-clip-free path; a matrix that is only
 * PSD up to rounding error should be repaired by the caller (e.g. by
 * shrinking off-diagonals toward 0) rather than silently clipped here.
 */
export function cholesky(matrix: readonly (readonly number[])[]): number[][] {
  const n = checkSquare(matrix);
  const L: number[][] = Array.from({ length: n }, () => Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = matrix[i]![j]!;
      for (let k = 0; k < j; k++) sum -= L[i]![k]! * L[j]![k]!;
      if (i === j) {
        if (sum < -PIVOT_TOL) {
          throw new RangeError(
            `matrix is not positive semidefinite: pivot ${i} is ${sum}`,
          );
        }
        L[i]![j] = sum <= PIVOT_TOL ? 0 : Math.sqrt(sum);
      } else if (L[j]![j] === 0) {
        // Zero pivot: the off-diagonal must also be (numerically) zero.
        if (Math.abs(sum) > PIVOT_TOL) {
          throw new RangeError(
            `matrix is not positive semidefinite: pivot ${j} is 0 but [${i}][${j}] residual is ${sum}`,
          );
        }
        L[i]![j] = 0;
      } else {
        L[i]![j] = sum / L[j]![j]!;
      }
    }
  }
  return L;
}

/** Multiply a lower-triangular matrix by a vector. */
function lowerMatVec(L: readonly (readonly number[])[], x: readonly number[]): number[] {
  return L.map((row, i) => {
    let s = 0;
    for (let k = 0; k <= i; k++) s += row[k]! * x[k]!;
    return s;
  });
}

/**
 * Correlated uniforms from independent uniforms, via a precomputed
 * Cholesky factor L of the correlation matrix.
 * `uniforms` must lie in (0, 1); the result is clamped to
 * [1e-15, 1 - 1e-15] so it stays a valid Metalog quantile argument.
 */
export function correlatedUniforms(
  L: readonly (readonly number[])[],
  uniforms: readonly number[],
): number[] {
  if (L.length !== uniforms.length) {
    throw new RangeError(`expected ${L.length} uniforms, got ${uniforms.length}`);
  }
  const z = uniforms.map((u) => {
    const zi = normalQuantile(u);
    if (Number.isNaN(zi)) throw new RangeError(`uniform must lie in (0, 1), got ${u}`);
    return zi;
  });
  const w = lowerMatVec(L, z);
  return w.map((wi) => {
    const v = normalCdf(wi);
    return Math.min(1 - 1e-15, Math.max(1e-15, v));
  });
}

/**
 * Gaussian copula over Metalog marginals.
 *
 * `correlation` is a d×d correlation matrix (unit diagonal, symmetric,
 * positive semidefinite) and `marginals` are d Metalog distributions,
 * in the same order. The factor L is computed once in the constructor.
 *
 * Sampling uses sipkit's HDR: dimension i draws
 * `hdrUniform(counter, { entity, varId: varId + i, seed3, seed4 })`,
 * so trials are reproducible from (counter, seeds) and dimensions are
 * independent before the copula couples them.
 */
export class GaussianCopula {
  readonly correlation: readonly (readonly number[])[];
  readonly marginals: readonly Metalog[];
  readonly factor: readonly (readonly number[])[];

  constructor(
    correlation: readonly (readonly number[])[],
    marginals: readonly Metalog[],
  ) {
    if (marginals.length === 0) throw new RangeError('at least one marginal is required');
    if (correlation.length !== marginals.length) {
      throw new RangeError(
        `correlation is ${correlation.length}×${correlation.length} but there are ${marginals.length} marginals`,
      );
    }
    for (let i = 0; i < marginals.length; i++) {
      const d = correlation[i]![i]!;
      if (Math.abs(d - 1) > 1e-10) {
        throw new RangeError(`correlation diagonal must be 1, got [${i}][${i}]=${d}`);
      }
    }
    this.correlation = correlation.map((row) => Object.freeze([...row]));
    this.marginals = Object.freeze([...marginals]);
    this.factor = cholesky(correlation);
  }

  /** Correlated uniforms for one trial. */
  sampleUniforms(counter: number, seeds: HdrSeeds): number[] {
    const uniforms = this.marginals.map((_, i) =>
      hdrUniform(counter, { ...seeds, varId: seeds.varId + i }),
    );
    return correlatedUniforms(this.factor, uniforms);
  }

  /** One joint sample: a value from each marginal, coupled by the copula. */
  sample(counter: number, seeds: HdrSeeds): number[] {
    const v = this.sampleUniforms(counter, seeds);
    return this.marginals.map((m, i) => m.quantile(v[i]!));
  }

  /** `count` joint samples, counters `firstCounter … firstCounter + count - 1`. */
  sampleMany(firstCounter: number, count: number, seeds: HdrSeeds): number[][] {
    const out: number[][] = [];
    for (let t = 0; t < count; t++) out.push(this.sample(firstCounter + t, seeds));
    return out;
  }
}
