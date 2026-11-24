//! Cholesky factorisation and the Gaussian copula, fitting sipkit's
//! SIPmath 3.1 kernel (HDR 2.0 PRNG + Metalog marginals).
//!
//! Recipe (the standard one):
//!
//!   1. `z_i = Φ⁻¹(u_i)` for independent uniforms `u_i` (here: HDR
//!      uniforms, one distinct `var_id` per dimension, same
//!      counter/entity/seeds).
//!   2. `w = L z`, where `L` is the lower Cholesky factor of the
//!      correlation matrix `R` (`R = L Lᵀ`).
//!   3. `v_i = Φ(w_i)` — correlated uniforms, each still Uniform(0, 1).
//!   4. `x_i = F_i⁻¹(v_i)` — here `F_i⁻¹` is a Metalog quantile, so the
//!      marginals are preserved exactly and only the dependence is
//!      Gaussian.
//!
//! No dependency is added: the Cholesky factor is ~40 lines, the
//! normal quantile/CDF live in [`crate::normal`]. A non-PSD matrix is
//! an error, never silently clipped — `ml-matrix`'s
//! `CholeskyDecomposition` was checked and returns a wrong factor for
//! `[[1, 2], [2, 1]]` instead of failing.

use std::fmt;

use crate::hdr::{HdrError, HdrSeeds, hdr_uniform};
use crate::metalog::Metalog;
use crate::normal::{normal_cdf, normal_quantile};

/// Smallest pivot treated as exactly zero (semidefinite case).
const PIVOT_TOL: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum CopulaError {
    /// Matrix is empty or a row has the wrong length.
    NotSquare {
        rows: usize,
        cols: usize,
    },
    NonFiniteEntry {
        row: usize,
        col: usize,
        value: f64,
    },
    NonSymmetric {
        row: usize,
        col: usize,
        a: f64,
        b: f64,
    },
    /// `pivot` is the (negative) residual at the failing diagonal.
    NotPositiveSemidefinite {
        index: usize,
        pivot: f64,
    },
    /// Correlation diagonal must be 1.
    NonUnitDiagonal {
        index: usize,
        value: f64,
    },
    /// Correlation size and marginal count differ.
    MarginalCountMismatch {
        matrix: usize,
        marginals: usize,
    },
    UniformCountMismatch {
        expected: usize,
        got: usize,
    },
    UniformOutOfRange {
        index: usize,
        value: f64,
    },
    Hdr(HdrError),
}

impl fmt::Display for CopulaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSquare { rows, cols } => {
                write!(f, "matrix must be square: {rows} rows, row length {cols}")
            }
            Self::NonFiniteEntry { row, col, value } => {
                write!(f, "matrix entry [{row}][{col}] must be finite, got {value}")
            }
            Self::NonSymmetric { row, col, a, b } => {
                write!(
                    f,
                    "matrix must be symmetric: [{row}][{col}]={a} != [{col}][{row}]={b}"
                )
            }
            Self::NotPositiveSemidefinite { index, pivot } => {
                write!(
                    f,
                    "matrix is not positive semidefinite: pivot {index} is {pivot}"
                )
            }
            Self::NonUnitDiagonal { index, value } => {
                write!(
                    f,
                    "correlation diagonal must be 1, got [{index}][{index}]={value}"
                )
            }
            Self::MarginalCountMismatch { matrix, marginals } => write!(
                f,
                "correlation is {matrix}×{matrix} but there are {marginals} marginals"
            ),
            Self::UniformCountMismatch { expected, got } => {
                write!(f, "expected {expected} uniforms, got {got}")
            }
            Self::UniformOutOfRange { index, value } => {
                write!(f, "uniform {index} must lie in (0, 1), got {value}")
            }
            Self::Hdr(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CopulaError {}

impl From<HdrError> for CopulaError {
    fn from(e: HdrError) -> Self {
        Self::Hdr(e)
    }
}

// Index loops are the readable form for matrix code here.
#[allow(clippy::needless_range_loop)]
fn check_square(matrix: &[Vec<f64>]) -> Result<usize, CopulaError> {
    let n = matrix.len();
    if n == 0 {
        return Err(CopulaError::NotSquare { rows: 0, cols: 0 });
    }
    for (i, row) in matrix.iter().enumerate() {
        if row.len() != n {
            return Err(CopulaError::NotSquare {
                rows: n,
                cols: row.len(),
            });
        }
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(CopulaError::NonFiniteEntry {
                    row: i,
                    col: j,
                    value: v,
                });
            }
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[i][j] - matrix[j][i]).abs() > 1e-10 {
                return Err(CopulaError::NonSymmetric {
                    row: i,
                    col: j,
                    a: matrix[i][j],
                    b: matrix[j][i],
                });
            }
        }
    }
    Ok(n)
}

/// Lower Cholesky factor `L` of a symmetric positive-semidefinite
/// matrix `A`, so `L Lᵀ = A` (within floating-point error).
///
/// Singular (semidefinite) matrices are accepted: a zero pivot yields
/// a zero row/column in `L`, which is the correct factor (e.g. ρ = 1
/// gives `z₂ = z₁`).
// Index loops are the readable form for the Cholesky recurrence here.
#[allow(clippy::needless_range_loop)]
pub fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, CopulaError> {
    let n = check_square(matrix)?;
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum < -PIVOT_TOL {
                    return Err(CopulaError::NotPositiveSemidefinite {
                        index: i,
                        pivot: sum,
                    });
                }
                l[i][j] = if sum <= PIVOT_TOL { 0.0 } else { sum.sqrt() };
            } else if l[j][j] == 0.0 {
                if sum.abs() > PIVOT_TOL {
                    return Err(CopulaError::NotPositiveSemidefinite {
                        index: j,
                        pivot: sum,
                    });
                }
                l[i][j] = 0.0;
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Lower Cholesky factor `L` of a symmetric positive-semidefinite matrix
/// given flat row-major, returned flat row-major. The flat shape is the
/// FFI-friendly form; [`cholesky`] above is the canonical one.
pub fn cholesky_flat(flat: &[f64], dim: usize) -> Result<Vec<f64>, CopulaError> {
    if dim == 0 || flat.len() != dim * dim {
        return Err(CopulaError::NotSquare {
            rows: dim,
            cols: flat.len(),
        });
    }
    let matrix: Vec<Vec<f64>> = flat.chunks(dim).map(<[f64]>::to_vec).collect();
    Ok(cholesky(&matrix)?.into_iter().flatten().collect())
}

fn lower_mat_vec(l: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    l.iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .take(i + 1)
                .zip(x.iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// Correlated uniforms from independent uniforms, via a precomputed
/// Cholesky factor `L` of the correlation matrix. `uniforms` must lie
/// in (0, 1); the result is clamped to `[1e-15, 1 - 1e-15]` so it
/// stays a valid Metalog quantile argument.
pub fn correlated_uniforms(l: &[Vec<f64>], uniforms: &[f64]) -> Result<Vec<f64>, CopulaError> {
    if l.len() != uniforms.len() {
        return Err(CopulaError::UniformCountMismatch {
            expected: l.len(),
            got: uniforms.len(),
        });
    }
    let mut z = Vec::with_capacity(uniforms.len());
    for (i, &u) in uniforms.iter().enumerate() {
        let zi = normal_quantile(u);
        if zi.is_nan() {
            return Err(CopulaError::UniformOutOfRange { index: i, value: u });
        }
        z.push(zi);
    }
    Ok(lower_mat_vec(l, &z)
        .iter()
        .map(|&w| normal_cdf(w).clamp(1e-15, 1.0 - 1e-15))
        .collect())
}

/// Gaussian copula over Metalog marginals.
///
/// `correlation` is a d×d correlation matrix (unit diagonal,
/// symmetric, positive semidefinite) and `marginals` are d Metalog
/// distributions, in the same order. The factor `L` is computed once
/// in the constructor.
///
/// Sampling uses sipkit's HDR: dimension `i` draws
/// `hdr_uniform(counter, { entity, var_id: var_id + i, seed3, seed4 })`,
/// so trials are reproducible from (counter, seeds) and dimensions
/// are independent before the copula couples them.
#[derive(Debug, Clone)]
pub struct GaussianCopula {
    correlation: Vec<Vec<f64>>,
    marginals: Vec<Metalog>,
    factor: Vec<Vec<f64>>,
}

impl GaussianCopula {
    pub fn new(correlation: Vec<Vec<f64>>, marginals: Vec<Metalog>) -> Result<Self, CopulaError> {
        if marginals.is_empty() {
            return Err(CopulaError::MarginalCountMismatch {
                matrix: correlation.len(),
                marginals: 0,
            });
        }
        let n = check_square(&correlation)?;
        if n != marginals.len() {
            return Err(CopulaError::MarginalCountMismatch {
                matrix: n,
                marginals: marginals.len(),
            });
        }
        for (i, row) in correlation.iter().enumerate() {
            if (row[i] - 1.0).abs() > 1e-10 {
                return Err(CopulaError::NonUnitDiagonal {
                    index: i,
                    value: row[i],
                });
            }
        }
        let factor = cholesky(&correlation)?;
        Ok(Self {
            correlation,
            marginals,
            factor,
        })
    }

    pub fn correlation(&self) -> &[Vec<f64>] {
        &self.correlation
    }

    pub fn marginals(&self) -> &[Metalog] {
        &self.marginals
    }

    pub fn factor(&self) -> &[Vec<f64>] {
        &self.factor
    }

    /// Correlated uniforms for one trial.
    pub fn sample_uniforms(&self, counter: u32, seeds: &HdrSeeds) -> Result<Vec<f64>, CopulaError> {
        let mut uniforms = Vec::with_capacity(self.marginals.len());
        for i in 0..self.marginals.len() {
            let dim_seeds = HdrSeeds {
                var_id: seeds
                    .var_id
                    .checked_add(i as u32)
                    .ok_or(HdrError::SeedOutOfRange {
                        name: "var_id",
                        value: u64::from(seeds.var_id) + i as u64,
                    })?,
                ..*seeds
            };
            uniforms.push(hdr_uniform(counter, &dim_seeds)?);
        }
        correlated_uniforms(&self.factor, &uniforms)
    }

    /// One joint sample: a value from each marginal, coupled by the copula.
    pub fn sample(&self, counter: u32, seeds: &HdrSeeds) -> Result<Vec<f64>, CopulaError> {
        let v = self.sample_uniforms(counter, seeds)?;
        Ok(self
            .marginals
            .iter()
            .zip(v.iter())
            .map(|(m, &vi)| m.quantile(vi))
            .collect())
    }

    /// `count` joint samples, counters `first_counter .. first_counter + count - 1`.
    pub fn sample_many(
        &self,
        first_counter: u32,
        count: usize,
        seeds: &HdrSeeds,
    ) -> Result<Vec<Vec<f64>>, CopulaError> {
        let mut out = Vec::with_capacity(count);
        for t in 0..count {
            out.push(self.sample(first_counter + t as u32, seeds)?);
        }
        Ok(out)
    }
}
