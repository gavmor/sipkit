//! Standard normal CDF / quantile (Wichura AS 241 quantile; A&S 7.1.26 CDF).
//!
//! Mirrors `packages/core/src/normal.ts` exactly — same coefficients,
//! same Abramowitz & Stegun 7.1.26 CDF — so the Rust and TS kernels
//! agree to the last digit the approximations allow.

/// Standard normal pdf.
pub fn normal_pdf(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF, via the error function
/// (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
pub fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    let erf = 1.0 - poly * (-ax * ax).exp();
    0.5 * (1.0 + s * erf)
}

// Wichura's Algorithm AS 241 (PPND16), Applied Statistics 37 (1988) 477-484.
// Coefficients are listed highest degree first (Horner order). Each
// denominator omits its constant term, which is 1. Transcribed from the
// published tables and diffed against R's `src/nmath/qnorm.c` (45 values,
// identical); `packages/core/src/normal.ts` carries the same literals.
//
// Regions, with q = p - 1/2:
//   |q| <= 0.425            central:      x = q * A(r) / B(r), r = 0.425^2 - q^2
//   |q| > 0.425, s <= 5     intermediate: x = C(s - 1.6) / D(s - 1.6)
//   |q| > 0.425, s > 5      extreme tail: x = E(s - 5)   / F(s - 5)
// where s = sqrt(-ln(min(p, 1 - p))). Absolute error is about 1e-16 relative
// over the double range of p down to ~1e-300.
const A: [f64; 8] = [
    2509.0809287301227,
    33430.57558358813,
    67265.7709270087,
    45921.95393154987,
    13731.69376550946,
    1971.5909503065513,
    133.14166789178438,
    3.3871328727963665,
];
const B: [f64; 7] = [
    5226.495278852854,
    28729.085735721943,
    39307.89580009271,
    21213.794301586597,
    5394.196021424751,
    687.1870074920579,
    42.31333070160091,
];
const C: [f64; 8] = [
    0.0007745450142783414,
    0.022723844989269184,
    0.2417807251774506,
    1.2704582524523684,
    3.6478483247632045,
    5.769497221460691,
    4.630337846156546,
    1.4234371107496835,
];
const D: [f64; 7] = [
    1.0507500716444169e-9,
    0.0005475938084995345,
    0.015198666563616457,
    0.14810397642748008,
    0.6897673349851,
    1.6763848301838038,
    2.053191626637759,
];
const E: [f64; 8] = [
    2.0103343992922881e-7,
    2.7115555687434876e-5,
    0.0012426609473880784,
    0.026532189526576124,
    0.29656057182850487,
    1.7848265399172913,
    5.463784911164114,
    6.657904643501103,
];
const F: [f64; 7] = [
    2.0442631033899397e-15,
    1.421511758316446e-7,
    1.8463183175100548e-5,
    0.0007868691311456133,
    0.014875361290850615,
    0.1369298809227358,
    0.599832206555888,
];
const SPLIT: f64 = 0.425;
const CENTRAL_OFFSET: f64 = 0.180625; // 0.425^2

/// Horner evaluation, highest-degree coefficient first.
fn horner(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |v, &c| v * x + c)
}

/// Standard normal quantile Φ⁻¹(p), Wichura's AS 241 (PPND16). Returns NaN
/// unless `0 < p < 1`, matching `Metalog::quantile`'s domain convention.
pub fn normal_quantile(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    let q = p - 0.5;
    if q.abs() <= SPLIT {
        let r = CENTRAL_OFFSET - q * q;
        return q * horner(&A, r) / (horner(&B, r) * r + 1.0);
    }
    let tail = if q < 0.0 { p } else { 1.0 - p };
    let s = (-tail.ln()).sqrt();
    let v = if s <= 5.0 {
        let y = s - 1.6;
        horner(&C, y) / (horner(&D, y) * y + 1.0)
    } else {
        let y = s - 5.0;
        horner(&E, y) / (horner(&F, y) * y + 1.0)
    };
    if q < 0.0 { -v } else { v }
}
