//! Standard normal CDF / quantile (Acklam, 2003).
//!
//! Mirrors `packages/core/src/normal.ts` exactly — same coefficients,
//! same Abramowitz & Stegun 7.1.26 CDF — so the Rust and TS kernels
//! agree to the last digit the approximations allow. The one
//! transcription trap is `A[2] = -2.759285104469687e2` (not
//! `…984…`); the wrong digit costs 1.75e-4 at p = 0.975.

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

// Acklam's rational approximation, |error| ~1.6e-9.
const A: [f64; 6] = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
];
const B: [f64; 5] = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
];
const C: [f64; 6] = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
];
const D: [f64; 4] = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
];
const P_LOW: f64 = 0.02425;
const P_HIGH: f64 = 1.0 - P_LOW;

fn horner(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().fold(0.0, |v, &c| v * x + c)
}

/// Standard normal quantile Φ⁻¹(p). Returns NaN unless `0 < p < 1`,
/// matching `Metalog::quantile`'s domain convention.
pub fn normal_quantile(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        return horner(&C, q) / (horner(&D, q) * q + 1.0);
    }
    if p > P_HIGH {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -horner(&C, q) / (horner(&D, q) * q + 1.0);
    }
    let q = p - 0.5;
    let r = q * q;
    horner(&A, r) * q / (horner(&B, r) * r + 1.0)
}
