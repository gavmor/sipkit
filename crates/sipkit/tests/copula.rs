use sipkit::copula::{GaussianCopula, cholesky, cholesky_flat, correlated_uniforms};
use sipkit::hdr::HdrSeeds;
use sipkit::metalog::{Metalog, MetalogBounds};
use sipkit::normal::{normal_cdf, normal_quantile};
use std::path::Path;

fn fixture_json(name: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/{name}"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn f64_vec(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn f64_mat(v: &serde_json::Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(f64_vec).collect()
}

fn bounds_of(b: &serde_json::Value) -> MetalogBounds {
    let num = |k: &str| b[k].as_f64().unwrap();
    match b["kind"].as_str().unwrap() {
        "unbounded" => MetalogBounds::Unbounded,
        "lower" => MetalogBounds::Lower(num("lower")),
        "upper" => MetalogBounds::Upper(num("upper")),
        "bounded" => MetalogBounds::Bounded {
            lower: num("lower"),
            upper: num("upper"),
        },
        other => panic!("unknown bounds kind {other}"),
    }
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| {
            (0..b[0].len())
                .map(|j| row.iter().zip(b.iter()).map(|(v, br)| v * br[j]).sum())
                .collect()
        })
        .collect()
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..a[0].len())
        .map(|j| a.iter().map(|row| row[j]).collect())
        .collect()
}

fn close(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(ra, rb)| ra.iter().zip(rb.iter()).all(|(x, y)| (x - y).abs() <= tol))
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..xs.len() {
        sxy += (xs[i] - mx) * (ys[i] - my);
        sxx += (xs[i] - mx).powi(2);
        syy += (ys[i] - my).powi(2);
    }
    sxy / (sxx * syy).sqrt()
}

#[test]
fn quantile_matches_known_points() {
    assert!((normal_quantile(0.5) - 0.0).abs() < 1e-12);
    assert!((normal_quantile(0.975) - 1.959963984540054).abs() < 5e-9);
    assert!((normal_quantile(0.025) + 1.959963984540054).abs() < 5e-9);
    assert!((normal_quantile(0.001) + 3.090232306167813).abs() < 1e-7);
    assert!((normal_quantile(0.999) - 3.090232306167813).abs() < 1e-7);
    for p in [0.0, 1.0, -0.1, 1.1, f64::NAN] {
        assert!(normal_quantile(p).is_nan());
    }
}

#[test]
fn quantile_round_trips_through_cdf() {
    for z in [-2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5] {
        assert!((normal_quantile(normal_cdf(z)) - z).abs() < 1e-13, "z={z}");
    }
}

#[test]
fn cholesky_factors_known_matrices() {
    let l = cholesky(&[vec![4.0, 2.0], vec![2.0, 3.0]]).unwrap();
    assert!((l[0][0] - 2.0).abs() < 1e-12);
    assert!((l[1][0] - 1.0).abs() < 1e-12);
    assert!((l[1][1] - std::f64::consts::SQRT_2).abs() < 1e-12);
    let l = cholesky(&[vec![1.0, 0.5], vec![0.5, 1.0]]).unwrap();
    assert!((l[1][0] - 0.5).abs() < 1e-12);
    assert!((l[1][1] - 0.75f64.sqrt()).abs() < 1e-12);
}

#[test]
fn cholesky_reconstructs_r() {
    let r = vec![
        vec![1.0, 0.8, 0.3],
        vec![0.8, 1.0, 0.5],
        vec![0.3, 0.5, 1.0],
    ];
    let l = cholesky(&r).unwrap();
    assert!(close(&mat_mul(&l, &transpose(&l)), &r, 1e-12));
}

#[test]
fn cholesky_accepts_singular_rho_1() {
    let r = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let l = cholesky(&r).unwrap();
    assert!(close(&mat_mul(&l, &transpose(&l)), &r, 1e-12));
}

#[test]
fn cholesky_rejects_bad_input() {
    assert!(cholesky(&[vec![1.0, 0.5], vec![0.6, 1.0]]).is_err());
    assert!(cholesky(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_err());
    assert!(cholesky(&[vec![1.0, 0.5]]).is_err());
    assert!(cholesky(&[]).is_err());
}

#[test]
fn cholesky_flat_matches_nested_cholesky() {
    let l = cholesky_flat(&[4.0, 2.0, 2.0, 3.0], 2).unwrap();
    assert_eq!(l.len(), 4);
    assert!((l[0] - 2.0).abs() < 1e-12);
    assert!((l[2] - 1.0).abs() < 1e-12);
    assert!((l[3] - std::f64::consts::SQRT_2).abs() < 1e-12);
    let r = vec![1.0, 0.8, 0.3, 0.8, 1.0, 0.5, 0.3, 0.5, 1.0];
    let l3 = cholesky_flat(&r, 3).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += l3[i * 3 + k] * l3[j * 3 + k];
            }
            assert!((s - r[i * 3 + j]).abs() < 1e-12, "({i}, {j})");
        }
    }
}

#[test]
fn cholesky_flat_rejects_bad_input() {
    assert!(cholesky_flat(&[1.0, 2.0, 2.0, 1.0], 2).is_err());
    assert!(cholesky_flat(&[1.0, 0.5], 2).is_err());
    assert!(cholesky_flat(&[], 0).is_err());
}

#[test]
fn copula_is_reproducible() {
    let seeds = HdrSeeds {
        entity: 1,
        var_id: 2,
        seed3: 3,
        seed4: 4,
    };
    let cop = GaussianCopula::new(
        vec![vec![1.0, 0.5], vec![0.5, 1.0]],
        vec![
            Metalog::new(vec![0.0, 1.0], MetalogBounds::Unbounded).unwrap(),
            Metalog::new(vec![10.0, 2.0], MetalogBounds::Unbounded).unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(
        cop.sample(7, &seeds).unwrap(),
        cop.sample(7, &seeds).unwrap()
    );
    assert_ne!(
        cop.sample(7, &seeds).unwrap(),
        cop.sample(8, &seeds).unwrap()
    );
}

#[test]
fn copula_recovers_target_correlation() {
    let seeds = HdrSeeds {
        entity: 1,
        var_id: 2,
        seed3: 3,
        seed4: 4,
    };
    let rho = 0.8;
    let cop = GaussianCopula::new(
        vec![vec![1.0, rho], vec![rho, 1.0]],
        vec![
            Metalog::new(vec![0.0, 1.0], MetalogBounds::Unbounded).unwrap(),
            Metalog::new(vec![0.0, 1.0], MetalogBounds::Unbounded).unwrap(),
        ],
    )
    .unwrap();
    let n = 20_000;
    let us: Vec<Vec<f64>> = (1..=n)
        .map(|c| cop.sample_uniforms(c, &seeds).unwrap())
        .collect();
    let xs: Vec<f64> = us.iter().map(|u| normal_quantile(u[0])).collect();
    let ys: Vec<f64> = us.iter().map(|u| normal_quantile(u[1])).collect();
    assert!(
        (pearson(&xs, &ys) - rho).abs() < 0.03,
        "r={}",
        pearson(&xs, &ys)
    );
}

#[test]
fn copula_preserves_marginals() {
    let seeds = HdrSeeds {
        entity: 1,
        var_id: 2,
        seed3: 3,
        seed4: 4,
    };
    let cop = GaussianCopula::new(
        vec![vec![1.0, 0.7], vec![0.7, 1.0]],
        vec![
            Metalog::new(vec![0.0, 1.0], MetalogBounds::Unbounded).unwrap(),
            // Lower-bounded Metalog works in log space: a1 = ln(median - lower).
            Metalog::new(vec![100.0f64.ln(), 0.2], MetalogBounds::Lower(0.0)).unwrap(),
        ],
    )
    .unwrap();
    let mut samples = cop.sample_many(1, 20_000, &seeds).unwrap();
    let mut xs: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let mut ys: Vec<f64> = samples.iter().map(|s| s[1]).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(xs[xs.len() / 2].abs() < 0.05);
    assert!((ys[ys.len() / 2] - 100.0).abs() < 1.5);
    samples.clear();
}

#[test]
fn copula_reduces_to_independence_at_rho_0() {
    let seeds = HdrSeeds {
        entity: 1,
        var_id: 2,
        seed3: 3,
        seed4: 4,
    };
    let cop = GaussianCopula::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![
            Metalog::new(vec![0.0, 1.0], MetalogBounds::Unbounded).unwrap(),
            Metalog::new(vec![0.0, 1.0], MetalogBounds::Unbounded).unwrap(),
        ],
    )
    .unwrap();
    let us: Vec<Vec<f64>> = (1..=20_000)
        .map(|c| cop.sample_uniforms(c, &seeds).unwrap())
        .collect();
    let xs: Vec<f64> = us.iter().map(|u| normal_quantile(u[0])).collect();
    let ys: Vec<f64> = us.iter().map(|u| normal_quantile(u[1])).collect();
    assert!(pearson(&xs, &ys).abs() < 0.03);
}

#[test]
fn copula_rejects_bad_construction() {
    let m = || Metalog::new(vec![0.0, 1.0], MetalogBounds::Unbounded).unwrap();
    assert!(GaussianCopula::new(vec![vec![2.0, 0.0], vec![0.0, 1.0]], vec![m(), m()]).is_err());
    assert!(GaussianCopula::new(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![m()]).is_err());
    let l = cholesky(&[vec![1.0, 0.5], vec![0.5, 1.0]]).unwrap();
    assert!(correlated_uniforms(&l, &[0.5]).is_err());
    assert!(correlated_uniforms(&l, &[0.0, 0.5]).is_err());
}

#[test]
fn copula_matches_shared_parity_fixture() {
    // The fixture is generated from this (canonical) implementation. The
    // Cholesky factor must match exactly (bit-identical f64); correlated
    // uniforms and samples pass through ln/exp, which other engines may
    // round one ulp apart, so they are compared within `value_tol`. The
    // same file is checked by the WASM suite.
    let json = fixture_json("copula-parity.json");
    let tol = json["value_tol"].as_f64().unwrap();
    let close = |actual: &[f64], expected: &[f64], what: &str| {
        assert_eq!(actual.len(), expected.len(), "{what} length");
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= tol * e.abs().max(1.0),
                "{what}: {a} vs {e}"
            );
        }
    };
    for case in json["cases"].as_array().unwrap() {
        let seeds = HdrSeeds {
            entity: case["seeds"]["entity"].as_u64().unwrap() as u32,
            var_id: case["seeds"]["varId"].as_u64().unwrap() as u32,
            seed3: case["seeds"]["seed3"].as_u64().unwrap() as u32,
            seed4: case["seeds"]["seed4"].as_u64().unwrap() as u32,
        };
        let marginals: Vec<Metalog> = case["marginals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Metalog::new(f64_vec(&m["a"]), bounds_of(&m["bounds"])).unwrap())
            .collect();
        let cop = GaussianCopula::new(f64_mat(&case["correlation"]), marginals).unwrap();
        assert_eq!(cop.factor().to_vec(), f64_mat(&case["factor"]));
        for s in case["samples"].as_array().unwrap() {
            let counter = s["counter"].as_u64().unwrap() as u32;
            close(
                &cop.sample_uniforms(counter, &seeds).unwrap(),
                &f64_vec(&s["uniforms"]),
                &format!("counter {counter} uniforms"),
            );
            close(
                &cop.sample(counter, &seeds).unwrap(),
                &f64_vec(&s["x"]),
                &format!("counter {counter} sample"),
            );
        }
    }
}

#[test]
fn normal_matches_accuracy_fixture() {
    // Tolerances come from the fixture itself; see its `measured` note
    // for how they were chosen (observed max error with ~4x headroom).
    let json = fixture_json("normal-accuracy.json");
    let qtol = json["quantile_tol"].as_f64().unwrap();
    let ctol = json["cdf_tol"].as_f64().unwrap();
    for q in json["quantile"].as_array().unwrap() {
        let (p, z) = (q["p"].as_f64().unwrap(), q["z"].as_f64().unwrap());
        let err = (normal_quantile(p) - z).abs();
        assert!(
            err <= qtol * z.abs().max(1.0),
            "p={p}: quantile error {err} exceeds {qtol} * max(1, |{z}|)"
        );
    }
    for c in json["cdf"].as_array().unwrap() {
        let (x, p) = (c["x"].as_f64().unwrap(), c["p"].as_f64().unwrap());
        let err = (normal_cdf(x) - p).abs();
        assert!(
            err <= ctol * p,
            "x={x}: cdf error {err} exceeds {ctol} * {p}"
        );
    }
}
