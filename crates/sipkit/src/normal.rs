//! Standard normal CDF / quantile. The quantile delegates to `distrs`
//! (Wichura AS 241); the CDF is Abramowitz & Stegun 7.1.26. The WASM bridge
//! calls these directly, so there is no second implementation to drift.

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

/// Standard normal quantile Φ⁻¹(p): Wichura's Algorithm AS 241 (PPND16),
/// Applied Statistics 37 (1988) 477-484, via [`distrs::Normal::ppf`].
///
/// Returns NaN unless `0 < p < 1`, matching `Metalog::quantile`'s domain
/// convention (`distrs` itself returns ±∞ at 0 and 1).
///
/// Measured against mpmath references (`fixtures/normal-accuracy.json`):
/// max scaled error 4.6e-16, in every branch down to p = 1e-300.
pub fn normal_quantile(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    distrs::Normal::ppf(p, 0.0, 1.0)
}
