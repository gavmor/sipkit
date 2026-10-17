//! Keelin (2016) metalog quantile functions with unbounded, semi-bounded and bounded variants.

use std::fmt;

/// Minimum number of metalog terms.
pub const MIN_TERMS: usize = 2;
/// Maximum number of metalog terms.
pub const MAX_TERMS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetalogBounds {
    Unbounded,
    Lower(f64),
    Upper(f64),
    Bounded { lower: f64, upper: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetalogError {
    /// Coefficient count outside `[MIN_TERMS, MAX_TERMS]`.
    TermCount(usize),
    NonFiniteCoefficient {
        index: usize,
        value: f64,
    },
    NonFiniteBound(f64),
    /// Bounded variant requires `lower < upper`.
    EmptyBounds {
        lower: f64,
        upper: f64,
    },
}

impl fmt::Display for MetalogError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TermCount(k) => write!(
                f,
                "metalog needs {MIN_TERMS} to {MAX_TERMS} coefficients, got {k}"
            ),
            Self::NonFiniteCoefficient { index, value } => {
                write!(f, "metalog coefficient a[{index}] = {value} is not finite")
            }
            Self::NonFiniteBound(b) => write!(f, "metalog bound {b} is not finite"),
            Self::EmptyBounds { lower, upper } => {
                write!(
                    f,
                    "metalog lower bound {lower} must be below upper bound {upper}"
                )
            }
        }
    }
}

impl std::error::Error for MetalogError {}

/// Validated metalog: `MIN_TERMS <= k <= MAX_TERMS` finite coefficients, finite bounds with
/// `lower < upper` when bounded.
#[derive(Debug, Clone, PartialEq)]
pub struct Metalog {
    a: Vec<f64>,
    bounds: MetalogBounds,
}

fn finite_bound(b: f64) -> Result<f64, MetalogError> {
    if b.is_finite() {
        Ok(b)
    } else {
        Err(MetalogError::NonFiniteBound(b))
    }
}

impl Metalog {
    pub fn new(coefficients: Vec<f64>, bounds: MetalogBounds) -> Result<Self, MetalogError> {
        let k = coefficients.len();
        if !(MIN_TERMS..=MAX_TERMS).contains(&k) {
            return Err(MetalogError::TermCount(k));
        }
        if let Some((index, &value)) = coefficients
            .iter()
            .enumerate()
            .find(|(_, a)| !a.is_finite())
        {
            return Err(MetalogError::NonFiniteCoefficient { index, value });
        }
        match bounds {
            MetalogBounds::Unbounded => {}
            MetalogBounds::Lower(b) | MetalogBounds::Upper(b) => {
                finite_bound(b)?;
            }
            MetalogBounds::Bounded { lower, upper } => {
                finite_bound(lower)?;
                finite_bound(upper)?;
                if lower >= upper {
                    return Err(MetalogError::EmptyBounds { lower, upper });
                }
            }
        }
        Ok(Self {
            a: coefficients,
            bounds,
        })
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.a
    }

    pub fn bounds(&self) -> &MetalogBounds {
        &self.bounds
    }

    /// Quantile at probability `y`; NaN unless `0 < y < 1`.
    pub fn quantile(&self, y: f64) -> f64 {
        if !(y > 0.0 && y < 1.0) {
            return f64::NAN;
        }
        let m = self.basis_sum(y);
        match self.bounds {
            MetalogBounds::Unbounded => m,
            MetalogBounds::Lower(lower) => lower + m.exp(),
            MetalogBounds::Upper(upper) => upper - (-m).exp(),
            MetalogBounds::Bounded { lower, upper } => lower + (upper - lower) / (1.0 + (-m).exp()),
        }
    }

    /// `M(y) = Σ a_j g_j(y)` with powers of `d = y − ½` built incrementally.
    #[inline]
    fn basis_sum(&self, y: f64) -> f64 {
        let a = &self.a;
        let logit = y.ln() - (-y).ln_1p();
        let d = y - 0.5;
        let mut m = a[0] + a[1] * logit;
        if let Some(&a3) = a.get(2) {
            m += a3 * d * logit;
        }
        if let Some(&a4) = a.get(3) {
            m += a4 * d;
        }
        let mut power = d;
        for pair in a.get(4..).unwrap_or_default().chunks(2) {
            power *= d;
            m += pair[0] * power;
            if let Some(&even) = pair.get(1) {
                m += even * power * logit;
            }
        }
        m
    }
}
