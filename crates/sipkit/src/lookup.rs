//! Discrete (empirical) lookup tables: inverse-CDF step functions over
//! cumulative-probability cutoffs (SIPmath Canonical Lib 06).
//!
//! A table is a list of `(cutoff, value)` pairs with cutoffs
//! non-decreasing in `[0, 1]`. The value of pair `i` covers
//! `u` in `(cutoff[i-1], cutoff[i]]` (with `cutoff[-1] = 0`): the first
//! pair whose cutoff is `>= u` wins, and `u` above every cutoff takes
//! the last value. `u` outside `[0, 1]` (including NaN) yields NaN,
//! matching [`crate::metalog::Metalog::quantile`]'s contract. An empty
//! table yields NaN; callers that need validation (non-empty, monotone
//! cutoffs) enforce it before calling.

/// Evaluate a discrete inverse CDF table at `u`.
pub fn eval_discrete_lookup(u: f64, table: &[(f64, f64)]) -> f64 {
    if !(0.0..=1.0).contains(&u) {
        return f64::NAN;
    }
    for &(cutoff, value) in table {
        if u <= cutoff {
            return value;
        }
    }
    table.last().map(|&(_, value)| value).unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &[(f64, f64)] = &[(0.25, 10.0), (0.5, 20.0), (1.0, 30.0)];

    #[test]
    fn lookup_intervals_are_left_open_right_closed() {
        assert_eq!(eval_discrete_lookup(0.0, TABLE), 10.0);
        assert_eq!(eval_discrete_lookup(0.25, TABLE), 10.0);
        assert_eq!(eval_discrete_lookup(0.250_000_1, TABLE), 20.0);
        assert_eq!(eval_discrete_lookup(0.5, TABLE), 20.0);
        assert_eq!(eval_discrete_lookup(0.75, TABLE), 30.0);
        assert_eq!(eval_discrete_lookup(1.0, TABLE), 30.0);
    }

    #[test]
    fn lookup_above_last_cutoff_takes_last_value() {
        assert_eq!(eval_discrete_lookup(0.9, &[(0.5, 5.0)]), 5.0);
    }

    #[test]
    fn lookup_out_of_range_is_nan() {
        for u in [-0.1, 1.1, f64::NAN] {
            assert!(eval_discrete_lookup(u, TABLE).is_nan(), "u={u}");
        }
        assert!(eval_discrete_lookup(0.5, &[]).is_nan());
    }
}
