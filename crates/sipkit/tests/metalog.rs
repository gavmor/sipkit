use sipkit::metalog::{Metalog, MetalogBounds, MetalogError};
use std::path::Path;

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

fn fixture_cases() -> Vec<(Metalog, Vec<(f64, f64)>)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/metalog-quantile.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    json["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let a = c["a"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let points = c["points"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| (p["y"].as_f64().unwrap(), p["x"].as_f64().unwrap()))
                .collect();
            (Metalog::new(a, bounds_of(&c["bounds"])).unwrap(), points)
        })
        .collect()
}

#[test]
fn quantile_matches_parity_vectors() {
    let cases = fixture_cases();
    assert!(!cases.is_empty());
    for (m, points) in cases {
        for (y, x) in points {
            let got = m.quantile(y);
            assert!(
                (got - x).abs() <= 1e-12 * x.abs().max(1.0),
                "k={} {:?} y={y}: got {got}, expected {x}",
                m.coefficients().len(),
                m.bounds()
            );
        }
    }
}

#[test]
fn quantile_is_nan_outside_open_unit_interval() {
    for (m, _) in fixture_cases() {
        for y in [
            0.0,
            1.0,
            -0.0,
            -1e-300,
            1.0 + f64::EPSILON,
            -1.0,
            2.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(m.quantile(y).is_nan(), "y={y}");
        }
    }
}

#[test]
fn outputs_respect_bounds() {
    let ys: Vec<f64> = [
        1e-300,
        1e-15,
        1e-9,
        0.3,
        0.5,
        0.7,
        1.0 - 1e-9,
        1.0 - f64::EPSILON / 2.0,
    ]
    .into_iter()
    .chain((1..1000).map(|i| f64::from(i) / 1000.0))
    .collect();
    for (m, _) in fixture_cases() {
        for &y in &ys {
            let x = m.quantile(y);
            assert!(!x.is_nan(), "y={y}");
            match *m.bounds() {
                MetalogBounds::Unbounded => {}
                MetalogBounds::Lower(lo) => assert!(x >= lo, "y={y} x={x}"),
                MetalogBounds::Upper(hi) => assert!(x <= hi, "y={y} x={x}"),
                MetalogBounds::Bounded { lower, upper } => {
                    assert!((lower..=upper).contains(&x), "y={y} x={x}")
                }
            }
        }
    }
}

#[test]
fn rejects_invalid_construction() {
    let ok = MetalogBounds::Unbounded;
    assert_eq!(Metalog::new(vec![1.0], ok), Err(MetalogError::TermCount(1)));
    assert_eq!(Metalog::new(vec![], ok), Err(MetalogError::TermCount(0)));
    assert_eq!(
        Metalog::new(vec![1.0; 17], ok),
        Err(MetalogError::TermCount(17))
    );
    assert!(Metalog::new(vec![1.0; 16], ok).is_ok());
    assert!(matches!(
        Metalog::new(vec![1.0, f64::NAN], ok),
        Err(MetalogError::NonFiniteCoefficient { index: 1, .. })
    ));
    assert!(matches!(
        Metalog::new(vec![f64::INFINITY, 1.0], ok),
        Err(MetalogError::NonFiniteCoefficient { index: 0, .. })
    ));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let a = || vec![0.0, 1.0];
        assert!(matches!(
            Metalog::new(a(), MetalogBounds::Lower(bad)),
            Err(MetalogError::NonFiniteBound(_))
        ));
        assert!(matches!(
            Metalog::new(a(), MetalogBounds::Upper(bad)),
            Err(MetalogError::NonFiniteBound(_))
        ));
        assert!(matches!(
            Metalog::new(
                a(),
                MetalogBounds::Bounded {
                    lower: 0.0,
                    upper: bad
                }
            ),
            Err(MetalogError::NonFiniteBound(_))
        ));
        assert!(matches!(
            Metalog::new(
                a(),
                MetalogBounds::Bounded {
                    lower: bad,
                    upper: 1.0
                }
            ),
            Err(MetalogError::NonFiniteBound(_))
        ));
    }
    for (lower, upper) in [(1.0, 1.0), (2.0, 1.0)] {
        assert_eq!(
            Metalog::new(vec![0.0, 1.0], MetalogBounds::Bounded { lower, upper }),
            Err(MetalogError::EmptyBounds { lower, upper })
        );
    }
}
