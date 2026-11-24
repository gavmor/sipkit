//! Standard normal CDF / quantile. The quantile delegates to `distrs`
//! (Wichura AS 241); the CDF is `libm::erfc`-based. The WASM bridge
//! calls these directly, so there is no second implementation to drift.

/// Standard normal pdf.
pub fn normal_pdf(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF: `Φ(x) = ½ · erfc(−x/√2)`, via [`libm::erfc`].
///
/// Going through `erfc` (not `1 + erf`) keeps full relative precision in
/// the left tail, where `1 + erf` cancels. Pure Rust, so it builds for
/// `wasm32-unknown-unknown`. NaN in, NaN out; ±∞ map to 1 and 0.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
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
