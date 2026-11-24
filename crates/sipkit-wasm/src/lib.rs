//! Batch WebAssembly bridge (`@gavmor/sipkit-wasm`) over the `sipkit` Rust kernel.
//!
//! Design goals:
//! - Zero per-trial JS <-> WASM FFI overhead: every endpoint operates on bulk
//!   slices (`&[f64]` / `&[u32]`) and returns `Vec<f64>` (marshaled by
//!   `wasm-bindgen` into a JS `Float64Array`).
//! - The copula math is not reimplemented here: `gaussianCopulaBatch` drives
//!   the kernel's [`correlated_uniforms`](sipkit::copula::correlated_uniforms)
//!   per trial, so batch output matches `GaussianCopula::sample_uniforms`
//!   exactly (including its `[1e-15, 1 - 1e-15]` clamp).
//! - Uses `u32` for HDR 2.0 seeds inside packed arrays because
//!   `MAX_SEED = 99_999_999 < u32::MAX`, keeping JS inputs as plain `number`s
//!   instead of `BigInt`. Scalar integer arguments are taken as `f64` and
//!   validated here: wasm-bindgen's integer coercion would silently truncate
//!   `1.5` and wrap `-1`.
//! - Column-major copula outputs so each copula layer is a contiguous
//!   `Float64Array.subarray(...)` slice ready for downstream `metalog_batch`.
//! - Exact SIPmath / Excel rounding semantics (Canonical Lib 08):
//!   `Up` rounds away from zero; `Down` rounds toward zero (`trunc`).

use sipkit::copula::{cholesky_flat, correlated_uniforms};
use sipkit::hdr::{HdrSeeds, MAX_SEED, hdr_fill, hdr_uniform};
use sipkit::lookup::eval_discrete_lookup;
use sipkit::metalog::{Metalog, MetalogBounds};
use wasm_bindgen::prelude::*;

/// Rounding modes defined by the SIPmath 3.1 specification (Canonical Lib 08).
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmRounding {
    /// No rounding applied.
    None = 0,
    /// Round to nearest integer, ties away from zero (matches Excel `ROUND`).
    Nearest = 1,
    /// Round away from zero (matches Excel `ROUNDUP`; e.g., -2.78 -> -3).
    Up = 2,
    /// Round toward zero (matches Excel `ROUNDDOWN`; e.g., -2.78 -> -2).
    Down = 3,
}

#[inline(always)]
fn apply_rounding(value: f64, mode: WasmRounding) -> f64 {
    if value.is_nan() || value.is_infinite() {
        return value;
    }
    match mode {
        WasmRounding::None => value,
        // Rust's f64::round rounds half-way cases away from 0.0 (Excel ROUND).
        WasmRounding::Nearest => value.round(),
        // Excel ROUNDUP: away from zero on both positive and negative values.
        WasmRounding::Up => {
            if value >= 0.0 {
                value.ceil()
            } else {
                value.floor()
            }
        }
        // Excel ROUNDDOWN: toward zero (truncation).
        WasmRounding::Down => value.trunc(),
    }
}

/// Streaming single-pass summary statistics using Chan's parallel Welford monoid.
/// Avoids allocating large trial buffers in JS when only summary moments are needed.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct WasmTrialStats {
    count: u32,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    nan_count: u32,
}

impl Default for WasmTrialStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            nan_count: 0,
        }
    }
}

impl WasmTrialStats {
    #[inline(always)]
    fn push(&mut self, x: f64) {
        if x.is_nan() {
            self.nan_count = self.nan_count.saturating_add(1);
            return;
        }
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.count as f64);
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    /// Associative Chan parallel merge (Monoid binary operation).
    pub fn merge_stats(self, other: Self) -> Self {
        if other.count == 0 {
            return Self {
                nan_count: self.nan_count + other.nan_count,
                ..self
            };
        }
        if self.count == 0 {
            return Self {
                nan_count: self.nan_count + other.nan_count,
                ..other
            };
        }

        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let n_ab = n_a + n_b;
        let delta = other.mean - self.mean;

        Self {
            count: self.count + other.count,
            mean: self.mean + delta * (n_b / n_ab),
            m2: self.m2 + other.m2 + delta * delta * (n_a * n_b / n_ab),
            min: self.min.min(other.min),
            max: self.max.max(other.max),
            nan_count: self.nan_count + other.nan_count,
        }
    }
}

#[wasm_bindgen]
impl WasmTrialStats {
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> u32 {
        self.count
    }

    #[wasm_bindgen(getter, js_name = "nanCount")]
    pub fn nan_count(&self) -> u32 {
        self.nan_count
    }

    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.mean }
    }

    /// Sample variance (Bessel-corrected, n - 1).
    #[wasm_bindgen(getter)]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            f64::NAN
        } else {
            self.m2 / ((self.count - 1) as f64)
        }
    }

    /// Sample standard deviation.
    #[wasm_bindgen(getter, js_name = "stdDev")]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    #[wasm_bindgen(getter)]
    pub fn min(&self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.min }
    }

    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f64 {
        if self.count == 0 { f64::NAN } else { self.max }
    }

    /// Merges another `WasmTrialStats` chunk into a new combined summary.
    pub fn merge(&self, other: &WasmTrialStats) -> WasmTrialStats {
        self.merge_stats(*other)
    }
}

// ---------------------------------------------------------------------------
// Internal validation helpers
// ---------------------------------------------------------------------------

/// `value` as `u32` if it is an integer in `[0, max]`.
fn int_arg(name: &str, value: f64, max: u32) -> Result<u32, JsValue> {
    if value.fract() == 0.0 && (0.0..=f64::from(max)).contains(&value) {
        Ok(value as u32)
    } else {
        Err(JsValue::from_str(&format!(
            "{name} must be an integer in [0, {max}], got {value}"
        )))
    }
}

fn seeds(entity: f64, var_id: f64, seed3: f64, seed4: f64) -> Result<HdrSeeds, JsValue> {
    Ok(HdrSeeds {
        entity: int_arg("entity", entity, MAX_SEED)?,
        var_id: int_arg("varId", var_id, MAX_SEED)?,
        seed3: int_arg("seed3", seed3, MAX_SEED)?,
        seed4: int_arg("seed4", seed4, MAX_SEED)?,
    })
}

/// Kernel errors surface as thrown `Error`s with the kernel's own message.
fn js_err<E: std::fmt::Display>(e: E) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn validate_counter_range(start_counter: u32, count: u32) -> Result<(), JsValue> {
    if count == 0 {
        return Ok(());
    }
    let last_counter = (start_counter as u64)
        .checked_add((count as u64) - 1)
        .ok_or_else(|| JsValue::from_str("Trial counter overflow"))?;
    if last_counter > u64::from(MAX_SEED) {
        return Err(JsValue::from_str(&format!(
            "Last trial counter ({last_counter}) exceeds MAX_SEED ({MAX_SEED})"
        )));
    }
    Ok(())
}

fn build_metalog(
    a_coeffs: &[f64],
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
) -> Result<Metalog, JsValue> {
    let bounds = match (lower_bound, upper_bound) {
        (Some(lower), Some(upper)) => MetalogBounds::Bounded { lower, upper },
        (Some(lower), None) => MetalogBounds::Lower(lower),
        (None, Some(upper)) => MetalogBounds::Upper(upper),
        (None, None) => MetalogBounds::Unbounded,
    };
    // Metalog::new is the authority on term count (2..=16), finite
    // coefficients, and finite bounds with lower < upper when bounded.
    Metalog::new(a_coeffs.to_vec(), bounds).map_err(js_err)
}

// ---------------------------------------------------------------------------
// 1. HDR 2.0 Batch Export
// ---------------------------------------------------------------------------

/// Generates `count` HDR 2.0 Uniform(0, 1) variates starting at `start_counter`
/// (typically `1` for 1-based `PM_Index`).
#[wasm_bindgen(js_name = "hdrBatch")]
pub fn hdr_batch(
    start_counter: f64,
    count: f64,
    entity: f64,
    var_id: f64,
    seed3: f64,
    seed4: f64,
) -> Result<Vec<f64>, JsValue> {
    let start_counter = int_arg("startCounter", start_counter, MAX_SEED)?;
    let count = int_arg("count", count, MAX_SEED + 1)?;
    validate_counter_range(start_counter, count)?;
    let mut out = vec![0.0; count as usize];
    hdr_fill(
        &seeds(entity, var_id, seed3, seed4)?,
        start_counter,
        &mut out,
    )
    .map_err(js_err)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// 2. Metalog Batch & Fused HDR -> Metalog Exports
// ---------------------------------------------------------------------------

/// Evaluates a Metalog quantile function over a precomputed uniform slice
/// (e.g., from a copula layer or empirical array). Values outside (0, 1) yield `NaN`.
#[wasm_bindgen(js_name = "metalogBatch")]
pub fn metalog_batch(
    uniforms: &[f64],
    a_coeffs: &[f64],
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    rounding: WasmRounding,
) -> Result<Vec<f64>, JsValue> {
    let dist = build_metalog(a_coeffs, lower_bound, upper_bound)?;
    let mut out = Vec::with_capacity(uniforms.len());
    for &u in uniforms {
        let q = dist.quantile(u);
        out.push(apply_rounding(q, rounding));
    }
    Ok(out)
}

/// Fused HDR 2.0 -> Metalog batch evaluation.
/// Avoids allocating or copying the intermediate Uniform(0, 1) buffer.
#[wasm_bindgen(js_name = "hdrMetalogBatch")]
#[allow(clippy::too_many_arguments)]
pub fn hdr_metalog_batch(
    start_counter: f64,
    count: f64,
    entity: f64,
    var_id: f64,
    seed3: f64,
    seed4: f64,
    a_coeffs: &[f64],
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    rounding: WasmRounding,
) -> Result<Vec<f64>, JsValue> {
    let start_counter = int_arg("startCounter", start_counter, MAX_SEED)?;
    let count = int_arg("count", count, MAX_SEED + 1)?;
    validate_counter_range(start_counter, count)?;
    let seeds = seeds(entity, var_id, seed3, seed4)?;
    let dist = build_metalog(a_coeffs, lower_bound, upper_bound)?;

    let mut out = Vec::with_capacity(count as usize);
    for offset in 0..count {
        let counter = start_counter + offset;
        let u = hdr_uniform(counter, &seeds).map_err(js_err)?;
        out.push(apply_rounding(dist.quantile(u), rounding));
    }
    Ok(out)
}

/// Streaming fused HDR 2.0 -> Metalog -> Welford summary statistics.
/// Allocates O(1) memory regardless of `count`.
#[wasm_bindgen(js_name = "hdrMetalogStats")]
#[allow(clippy::too_many_arguments)]
pub fn hdr_metalog_stats(
    start_counter: f64,
    count: f64,
    entity: f64,
    var_id: f64,
    seed3: f64,
    seed4: f64,
    a_coeffs: &[f64],
    lower_bound: Option<f64>,
    upper_bound: Option<f64>,
    rounding: WasmRounding,
) -> Result<WasmTrialStats, JsValue> {
    let start_counter = int_arg("startCounter", start_counter, MAX_SEED)?;
    let count = int_arg("count", count, MAX_SEED + 1)?;
    validate_counter_range(start_counter, count)?;
    let seeds = seeds(entity, var_id, seed3, seed4)?;
    let dist = build_metalog(a_coeffs, lower_bound, upper_bound)?;

    let mut stats = WasmTrialStats::default();
    for offset in 0..count {
        let counter = start_counter + offset;
        let u = hdr_uniform(counter, &seeds).map_err(js_err)?;
        stats.push(apply_rounding(dist.quantile(u), rounding));
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// 3. Discrete Lookup Table Batch Exports (Canonical Lib 06)
// ---------------------------------------------------------------------------

fn validate_lookup_table(cutoffs: &[f64], values: &[f64]) -> Result<Vec<(f64, f64)>, JsValue> {
    if cutoffs.is_empty() || cutoffs.len() != values.len() {
        return Err(JsValue::from_str(
            "Lookup table cutoffs and values must be non-empty and have equal length",
        ));
    }
    let mut prev = 0.0;
    let mut table = Vec::with_capacity(cutoffs.len());
    for (&p, &val) in cutoffs.iter().zip(values.iter()) {
        if p.is_nan() || p < prev || p > 1.0 + 1e-12 {
            return Err(JsValue::from_str(
                "Lookup table cumulative probabilities must be monotonically non-decreasing in [0, 1]",
            ));
        }
        prev = p;
        table.push((p, val));
    }
    Ok(table)
}

/// Evaluates an empirical discrete inverse CDF table over a uniform slice.
#[wasm_bindgen(js_name = "lookupBatch")]
pub fn lookup_batch(
    uniforms: &[f64],
    cutoffs: &[f64],
    values: &[f64],
) -> Result<Vec<f64>, JsValue> {
    let table = validate_lookup_table(cutoffs, values)?;
    let mut out = Vec::with_capacity(uniforms.len());
    for &u in uniforms {
        out.push(eval_discrete_lookup(u, &table));
    }
    Ok(out)
}

/// Fused HDR 2.0 -> Discrete Lookup Table batch evaluation.
#[wasm_bindgen(js_name = "hdrLookupBatch")]
#[allow(clippy::too_many_arguments)]
pub fn hdr_lookup_batch(
    start_counter: f64,
    count: f64,
    entity: f64,
    var_id: f64,
    seed3: f64,
    seed4: f64,
    cutoffs: &[f64],
    values: &[f64],
) -> Result<Vec<f64>, JsValue> {
    let start_counter = int_arg("startCounter", start_counter, MAX_SEED)?;
    let count = int_arg("count", count, MAX_SEED + 1)?;
    validate_counter_range(start_counter, count)?;
    let seeds = seeds(entity, var_id, seed3, seed4)?;
    let table = validate_lookup_table(cutoffs, values)?;

    let mut out = Vec::with_capacity(count as usize);
    for offset in 0..count {
        let counter = start_counter + offset;
        let u = hdr_uniform(counter, &seeds).map_err(js_err)?;
        out.push(eval_discrete_lookup(u, &table));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 4. Cholesky & Gaussian Copula Batch Exports (Canonical Lib 02)
// ---------------------------------------------------------------------------

/// Computes the lower-triangular Cholesky factor `L` (row-major `dim * dim`)
/// of a symmetric positive-definite correlation matrix.
/// Explicitly rejects asymmetric or non-positive-definite matrices.
#[wasm_bindgen(js_name = "choleskyDecompose")]
pub fn cholesky_decompose(dim: f64, corr_flat: &[f64]) -> Result<Vec<f64>, JsValue> {
    let dim = int_arg("dim", dim, 10_000)? as usize;
    if dim == 0 {
        return Err(JsValue::from_str(
            "Correlation matrix dimension must be > 0",
        ));
    }
    if corr_flat.len() != dim * dim {
        return Err(JsValue::from_str(&format!(
            "Expected {} elements for {}x{} matrix, got {}",
            dim * dim,
            dim,
            dim,
            corr_flat.len()
        )));
    }

    // Explicit symmetry check so invalid inputs never silently factor the lower triangle
    for i in 0..dim {
        for j in (i + 1)..dim {
            let a = corr_flat[i * dim + j];
            let b = corr_flat[j * dim + i];
            if a.is_nan() || b.is_nan() || (a - b).abs() > 1e-12 {
                return Err(JsValue::from_str(&format!(
                    "Correlation matrix is not symmetric at ({i}, {j}): {a} != {b}"
                )));
            }
        }
    }

    cholesky_flat(corr_flat, dim).map_err(js_err)
}

/// Evaluates `count` trials of a `dim`-dimensional Gaussian copula driven by HDR 2.0 RNGs.
///
/// Arguments:
/// - `seeds_flat`: packed length `dim * 4`, each row `[entity, varId, seed3, seed4]`
/// - `cholesky_l`: row-major lower-triangular factor `L` of length `dim * dim`
///
/// Returns a **column-major** `Float64Array` of length `dim * count`:
/// layer `d` occupies `[d * count .. (d + 1) * count]`, enabling zero-copy
/// `subarray()` slicing per downstream SIP in TypeScript.
///
/// The per-trial math is the kernel's `correlated_uniforms` (probit, `L z`,
/// CDF, clamped to `[1e-15, 1 - 1e-15]`), so batch output matches
/// `GaussianCopula::sample_uniforms` exactly.
#[wasm_bindgen(js_name = "gaussianCopulaBatch")]
pub fn gaussian_copula_batch(
    dim: f64,
    start_counter: f64,
    count: f64,
    seeds_flat: &[u32],
    cholesky_l: &[f64],
) -> Result<Vec<f64>, JsValue> {
    let dim = int_arg("dim", dim, 10_000)? as usize;
    let start_counter = int_arg("startCounter", start_counter, MAX_SEED)?;
    let count = int_arg("count", count, MAX_SEED + 1)?;
    if dim == 0 {
        return Err(JsValue::from_str("Copula dimension must be > 0"));
    }
    validate_counter_range(start_counter, count)?;
    if seeds_flat.len() != dim * 4 {
        return Err(JsValue::from_str(&format!(
            "Expected {} packed seed values (dim * 4), got {}",
            dim * 4,
            seeds_flat.len()
        )));
    }
    if cholesky_l.len() != dim * dim {
        return Err(JsValue::from_str(&format!(
            "Expected {} Cholesky elements (dim * dim), got {}",
            dim * dim,
            cholesky_l.len()
        )));
    }

    // Validate all seeds up-front (packed values are already u32; only the
    // MAX_SEED range needs checking).
    let mut seeds = Vec::with_capacity(dim);
    for d in 0..dim {
        let base = d * 4;
        for (label, &value) in ["entity", "varId", "seed3", "seed4"]
            .iter()
            .zip(&seeds_flat[base..base + 4])
        {
            if value > MAX_SEED {
                return Err(JsValue::from_str(&format!(
                    "HDR seed '{label}' ({value}) exceeds MAX_SEED ({MAX_SEED})"
                )));
            }
        }
        seeds.push(HdrSeeds {
            entity: seeds_flat[base],
            var_id: seeds_flat[base + 1],
            seed3: seeds_flat[base + 2],
            seed4: seeds_flat[base + 3],
        });
    }

    let l_nested: Vec<Vec<f64>> = cholesky_l.chunks(dim).map(<[f64]>::to_vec).collect();
    let n = count as usize;
    let mut out_col_major = vec![0.0_f64; dim * n];
    let mut uniforms = vec![0.0_f64; dim];

    for t in 0..n {
        let counter = start_counter + t as u32;
        for d in 0..dim {
            uniforms[d] = hdr_uniform(counter, &seeds[d]).map_err(js_err)?;
        }
        let correlated = correlated_uniforms(&l_nested, &uniforms).map_err(js_err)?;
        for i in 0..dim {
            out_col_major[i * n + t] = correlated[i];
        }
    }

    Ok(out_col_major)
}

// ---------------------------------------------------------------------------
// 5. Scalar normal functions (kept for the accuracy fixture and callers
//    that need a single value without a batch)
// ---------------------------------------------------------------------------

/// Standard normal quantile Φ⁻¹(p); NaN unless `0 < p < 1`.
#[wasm_bindgen(js_name = "normalQuantile")]
pub fn normal_quantile(p: f64) -> f64 {
    sipkit::normal::normal_quantile(p)
}

/// Standard normal CDF Φ(x).
#[wasm_bindgen(js_name = "normalCdf")]
pub fn normal_cdf(x: f64) -> f64 {
    sipkit::normal::normal_cdf(x)
}

// ---------------------------------------------------------------------------
// 6. Generic Trial Summary Helper
// ---------------------------------------------------------------------------

/// Computes single-pass Welford statistics over any `Float64Array` of trials.
#[wasm_bindgen(js_name = "summarizeTrials")]
pub fn summarize_trials(trials: &[f64]) -> WasmTrialStats {
    let mut stats = WasmTrialStats::default();
    for &x in trials {
        stats.push(x);
    }
    stats
}
