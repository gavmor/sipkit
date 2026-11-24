/**
 * Standard normal CDF / quantile (Acklam, 2003).
 *
 * Prior art in JS/TS: the pieces exist separately, not as one
 * SIPmath-style copula — `ts-gaussian` is univariate only (pdf/cdf/ppf),
 * `@chaos-labs/multivariate-normal` is a pure-JS port of NumPy's
 * `random.multivariate_normal` (Cholesky inside, Math.random-seeded,
 * no HDR, no Metalog marginals), `ml-matrix` and JSR `@maths/matrix`
 * expose a Cholesky factorisation but no copula, and Symi's
 * `gaussianCopula` is not exposed by its WASM/JS bindings.
 * Hence this file: ~90 lines, zero dependencies, matching sipkit's
 * style (plain `number` arithmetic, RangeError on out-of-domain input,
 * NaN outside (0, 1) for the quantile, as `Metalog.quantile` does).
 */

/** Standard normal pdf. */
export function normalPdf(x: number): number {
  if (!Number.isFinite(x)) return 0;
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Standard normal CDF, via the error function
 * (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
 */
export function normalCdf(x: number): number {
  if (Number.isNaN(x)) return NaN;
  if (x === Infinity) return 1;
  if (x === -Infinity) return 0;
  // erf(x / sqrt(2)) via A&S 7.1.26
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const erf = 1 - poly * Math.exp(-ax * ax);
  return 0.5 * (1 + s * erf);
}

// Acklam's rational approximation, |error| < 1.15e-9 in absolute value.
const A = [
  -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
  1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
];
const B = [
  -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
  6.680131188771972e1, -1.328068155288572e1,
];
const C = [
  -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
  -2.549732539343734, 4.374664141464968, 2.938163982698783,
];
const D = [
  7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416,
];
const P_LOW = 0.02425;
const P_HIGH = 1 - P_LOW;

/**
 * Standard normal quantile Φ⁻¹(p). Returns NaN unless 0 < p < 1,
 * matching `Metalog.quantile`'s domain convention.
 */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) return NaN;
  let q: number;
  let r: number;
  if (p < P_LOW) {
    // Lower tail.
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((C[0]! * q + C[1]!) * q + C[2]!) * q + C[3]!) * q + C[4]!) * q + C[5]!) /
      ((((D[0]! * q + D[1]!) * q + D[2]!) * q + D[3]!) * q + 1)
    );
  }
  if (p > P_HIGH) {
    // Upper tail.
    q = Math.sqrt(-2 * Math.log(1 - p));
    return (
      -(((((C[0]! * q + C[1]!) * q + C[2]!) * q + C[3]!) * q + C[4]!) * q + C[5]!) /
      ((((D[0]! * q + D[1]!) * q + D[2]!) * q + D[3]!) * q + 1)
    );
  }
  // Central region.
  q = p - 0.5;
  r = q * q;
  return (
    (((((A[0]! * r + A[1]!) * r + A[2]!) * r + A[3]!) * r + A[4]!) * r + A[5]!) * q /
    (((((B[0]! * r + B[1]!) * r + B[2]!) * r + B[3]!) * r + B[4]!) * r + 1)
  );
}
