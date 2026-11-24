/**
 * Standard normal CDF / quantile (Wichura AS 241 quantile; A&S 7.1.26 CDF).
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

// Wichura's Algorithm AS 241 (PPND16), Applied Statistics 37 (1988) 477-484.
// Same literals, same Horner order and same operation order as
// `crates/sipkit/src/normal.rs`. Coefficients are highest degree first;
// each denominator omits its constant term, which is 1. Regions, with
// q = p - 1/2 and s = sqrt(-ln(min(p, 1 - p))):
//   |q| <= 0.425            x = q * A(r) / B(r), r = 0.425^2 - q^2
//   |q| > 0.425, s <= 5     x = C(s - 1.6) / D(s - 1.6)
//   |q| > 0.425, s > 5      x = E(s - 5)   / F(s - 5)
const A = [
  2509.0809287301227,
  33430.57558358813,
  67265.7709270087,
  45921.95393154987,
  13731.69376550946,
  1971.5909503065513,
  133.14166789178438,
  3.3871328727963665,
];
const B = [
  5226.495278852854,
  28729.085735721943,
  39307.89580009271,
  21213.794301586597,
  5394.196021424751,
  687.1870074920579,
  42.31333070160091,
];
const C = [
  0.0007745450142783414,
  0.022723844989269184,
  0.2417807251774506,
  1.2704582524523684,
  3.6478483247632045,
  5.769497221460691,
  4.630337846156546,
  1.4234371107496835,
];
const D = [
  1.0507500716444169e-9,
  0.0005475938084995345,
  0.015198666563616457,
  0.14810397642748008,
  0.6897673349851,
  1.6763848301838038,
  2.053191626637759,
];
const E = [
  2.0103343992922881e-7,
  2.7115555687434876e-5,
  0.0012426609473880784,
  0.026532189526576124,
  0.29656057182850487,
  1.7848265399172913,
  5.463784911164114,
  6.657904643501103,
];
const F = [
  2.0442631033899397e-15,
  1.421511758316446e-7,
  1.8463183175100548e-5,
  0.0007868691311456133,
  0.014875361290850615,
  0.1369298809227358,
  0.599832206555888,
];
const SPLIT = 0.425;
const CENTRAL_OFFSET = 0.180625; // 0.425^2

/** Horner evaluation, highest-degree coefficient first. */
function horner(coeffs: readonly number[], x: number): number {
  let v = 0;
  for (const c of coeffs) v = v * x + c;
  return v;
}

/**
 * Standard normal quantile Φ⁻¹(p), Wichura's AS 241 (PPND16). Returns NaN
 * unless 0 < p < 1, matching `Metalog.quantile`'s domain convention.
 */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) return NaN;
  const q = p - 0.5;
  if (Math.abs(q) <= SPLIT) {
    const r = CENTRAL_OFFSET - q * q;
    return (q * horner(A, r)) / (horner(B, r) * r + 1);
  }
  const tail = q < 0 ? p : 1 - p;
  const s = Math.sqrt(-Math.log(tail));
  let v: number;
  if (s <= 5) {
    const y = s - 1.6;
    v = horner(C, y) / (horner(D, y) * y + 1);
  } else {
    const y = s - 5;
    v = horner(E, y) / (horner(F, y) * y + 1);
  }
  return q < 0 ? -v : v;
}
