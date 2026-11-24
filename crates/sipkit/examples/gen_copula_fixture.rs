//! Generator for `fixtures/copula-parity.json`: runs the Rust kernel
//! (the canonical implementation) and prints the fixture.
//!
//! Regenerate with:
//!   cargo run -p sipkit --example gen_copula_fixture > fixtures/copula-parity.json
//! Re-running must reproduce the committed fixture byte-identically;
//! if it does not, either the kernel changed or the fixture was hand-edited.

use serde_json::json;
use sipkit::copula::GaussianCopula;
use sipkit::hdr::HdrSeeds;
use sipkit::metalog::{Metalog, MetalogBounds};

fn case(
    correlation: Vec<Vec<f64>>,
    marginals: Vec<(Vec<f64>, MetalogBounds)>,
    counters: &[u32],
) -> serde_json::Value {
    let seeds = HdrSeeds {
        entity: 1,
        var_id: 2,
        seed3: 3,
        seed4: 4,
    };
    let ms: Vec<Metalog> = marginals
        .iter()
        .map(|(a, b)| Metalog::new(a.clone(), *b).unwrap())
        .collect();
    let cop = GaussianCopula::new(correlation.clone(), ms).unwrap();
    let samples: Vec<serde_json::Value> = counters
        .iter()
        .map(|&c| {
            json!({
                "counter": c,
                "uniforms": cop.sample_uniforms(c, &seeds).unwrap(),
                "x": cop.sample(c, &seeds).unwrap(),
            })
        })
        .collect();
    let marg_json: Vec<serde_json::Value> = marginals
        .iter()
        .map(|(a, b)| {
            let kind = match b {
                MetalogBounds::Unbounded => "unbounded",
                MetalogBounds::Lower(_) => "lower",
                MetalogBounds::Upper(_) => "upper",
                MetalogBounds::Bounded { .. } => "bounded",
            };
            json!({"a": a, "bounds": {"kind": kind}})
        })
        .collect();
    json!({
        "correlation": correlation,
        "marginals": marg_json,
        "seeds": {"entity": 1, "varId": 2, "seed3": 3, "seed4": 4},
        "factor": cop.factor(),
        "samples": samples,
    })
}

fn main() {
    let unbounded = MetalogBounds::Unbounded;
    let fixture = json!({
        "description": "Gaussian copula parity vectors. Generated from the Rust kernel (the canonical implementation); the WASM bridge must reproduce them. Cholesky factors must match exactly (bit-identical f64: only +, -, *, / and sqrt, which IEEE 754 rounds exactly). Correlated uniforms and samples pass through ln and exp, which Rust libm and WASM may round one ulp apart, so they are compared within value_tol. The WASM bridge calls the same kernel functions, so drift fails here by orders of magnitude, not by one ulp.",
        "value_tol": 1e-15,
        "value_tol_rule": "|actual - expected| <= value_tol * max(1, |expected|), for correlated uniforms and samples; factors are compared exactly",
        "source": "Generated from the Rust kernel (the canonical implementation); regenerate with `cargo run -p sipkit --example gen_copula_fixture > fixtures/copula-parity.json`.",
        "cases": [
            case(
                vec![vec![1.0, 0.5], vec![0.5, 1.0]],
                vec![(vec![0.0, 1.0], unbounded), (vec![10.0, 2.0], unbounded)],
                &[1, 2, 7, 8, 100, 1000],
            ),
            case(
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![(vec![0.0, 1.0], unbounded), (vec![10.0, 2.0], unbounded)],
                &[7, 42],
            ),
            case(
                vec![
                    vec![1.0, 0.5, 0.2],
                    vec![0.5, 1.0, 0.3],
                    vec![0.2, 0.3, 1.0],
                ],
                vec![
                    (vec![0.0, 1.0], unbounded),
                    (vec![10.0, 2.0], unbounded),
                    (vec![-5.0, 0.5], unbounded),
                ],
                &[7, 8],
            ),
        ],
    });
    println!("{}", serde_json::to_string_pretty(&fixture).unwrap());
}
