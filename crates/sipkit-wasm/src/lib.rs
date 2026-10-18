//! JavaScript bindings for the SIPmath kernel. Domain errors surface as thrown `Error`s.
//!
//! Integer arguments are taken as `f64` and validated here: wasm-bindgen's `u32` coercion would
//! silently truncate `1.5` and wrap `-1`, diverging from `@sipkit/core`.

use sipkit::hdr::{self, HdrSeeds, MAX_SEED};
use sipkit::metalog::{self, MetalogBounds};
use wasm_bindgen::prelude::*;

/// `value` as `u32` if it is an integer in `[0, max]`.
fn int_arg(name: &str, value: f64, max: u32) -> Result<u32, JsError> {
    if value.fract() == 0.0 && (0.0..=f64::from(max)).contains(&value) {
        Ok(value as u32)
    } else {
        Err(JsError::new(&format!(
            "{name} must be an integer in [0, {max}], got {value}"
        )))
    }
}

fn seeds(entity: f64, var_id: f64, seed3: f64, seed4: f64) -> Result<HdrSeeds, JsError> {
    Ok(HdrSeeds {
        entity: int_arg("entity", entity, MAX_SEED)?,
        var_id: int_arg("varId", var_id, MAX_SEED)?,
        seed3: int_arg("seed3", seed3, MAX_SEED)?,
        seed4: int_arg("seed4", seed4, MAX_SEED)?,
    })
}

fn fill(seeds: &HdrSeeds, first_counter: f64, count: f64) -> Result<Vec<f64>, JsError> {
    let first_counter = int_arg("firstCounter", first_counter, MAX_SEED)?;
    let count = int_arg("count", count, MAX_SEED + 1)?;
    let mut out = vec![0.0; count as usize];
    hdr::hdr_fill(seeds, first_counter, &mut out)?;
    Ok(out)
}

/// HDR 2.0 uniform for trial `counter`; every input must be an integer in `[0, MAX_SEED]`.
#[wasm_bindgen(js_name = hdrUniform)]
pub fn hdr_uniform(
    counter: f64,
    entity: f64,
    #[wasm_bindgen(js_name = varId)] var_id: f64,
    seed3: f64,
    seed4: f64,
) -> Result<f64, JsError> {
    Ok(hdr::hdr_uniform(
        int_arg("counter", counter, MAX_SEED)?,
        &seeds(entity, var_id, seed3, seed4)?,
    )?)
}

/// HDR 2.0 uniforms for counters `firstCounter .. firstCounter + count`.
#[wasm_bindgen(js_name = hdrFill)]
pub fn hdr_fill(
    entity: f64,
    #[wasm_bindgen(js_name = varId)] var_id: f64,
    seed3: f64,
    seed4: f64,
    #[wasm_bindgen(js_name = firstCounter)] first_counter: f64,
    count: f64,
) -> Result<Vec<f64>, JsError> {
    fill(&seeds(entity, var_id, seed3, seed4)?, first_counter, count)
}

/// Metalog quantile function; `lower`/`upper` select the bounds variant.
#[wasm_bindgen]
pub struct Metalog(metalog::Metalog);

#[wasm_bindgen]
impl Metalog {
    #[wasm_bindgen(constructor)]
    pub fn new(
        coefficients: Vec<f64>,
        lower: Option<f64>,
        upper: Option<f64>,
    ) -> Result<Metalog, JsError> {
        let bounds = match (lower, upper) {
            (None, None) => MetalogBounds::Unbounded,
            (Some(lower), None) => MetalogBounds::Lower(lower),
            (None, Some(upper)) => MetalogBounds::Upper(upper),
            (Some(lower), Some(upper)) => MetalogBounds::Bounded { lower, upper },
        };
        Ok(Self(metalog::Metalog::new(coefficients, bounds)?))
    }

    /// Quantile at `y`; NaN unless `0 < y < 1`.
    pub fn quantile(&self, y: f64) -> f64 {
        self.0.quantile(y)
    }

    /// Elementwise `quantile` over `ys`.
    pub fn quantiles(&self, ys: &[f64]) -> Vec<f64> {
        ys.iter().map(|&y| self.0.quantile(y)).collect()
    }

    /// `quantile(hdrUniform(c, ...))` for `c` in `firstCounter .. firstCounter + count`.
    pub fn simulate(
        &self,
        entity: f64,
        #[wasm_bindgen(js_name = varId)] var_id: f64,
        seed3: f64,
        seed4: f64,
        #[wasm_bindgen(js_name = firstCounter)] first_counter: f64,
        count: f64,
    ) -> Result<Vec<f64>, JsError> {
        let mut out = fill(&seeds(entity, var_id, seed3, seed4)?, first_counter, count)?;
        for v in &mut out {
            *v = self.0.quantile(*v);
        }
        Ok(out)
    }
}
