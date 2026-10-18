use sipkit::hdr::{HdrError, HdrSeeds, MAX_SEED, hdr_fill, hdr_uniform};
use std::path::Path;

fn golden() -> (HdrSeeds, Vec<(u32, f64)>) {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/hdr-2.0-seed-standard.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let seed = |k: &str| json["seeds"][k].as_u64().unwrap() as u32;
    let seeds = HdrSeeds {
        entity: seed("entity"),
        var_id: seed("varId"),
        seed3: seed("seed3"),
        seed4: seed("seed4"),
    };
    let cases = json["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["counter"].as_u64().unwrap() as u32,
                c["u"].as_f64().unwrap(),
            )
        })
        .collect();
    (seeds, cases)
}

#[test]
fn uniform_matches_seed_standard_exactly() {
    let (seeds, cases) = golden();
    assert_eq!(cases.len(), 10_000);
    for (counter, u) in cases {
        assert_eq!(
            hdr_uniform(counter, &seeds).unwrap(),
            u,
            "counter {counter}"
        );
    }
}

#[test]
fn fill_matches_seed_standard_exactly() {
    let (seeds, cases) = golden();
    let first = cases[0].0;
    assert!(
        cases
            .iter()
            .enumerate()
            .all(|(i, &(c, _))| c == first + i as u32)
    );
    let mut out = vec![0.0; cases.len()];
    hdr_fill(&seeds, first, &mut out).unwrap();
    for ((counter, u), got) in cases.into_iter().zip(out) {
        assert_eq!(got, u, "counter {counter}");
    }
}

#[test]
fn extreme_inputs_stay_in_open_unit_interval() {
    let max = HdrSeeds {
        entity: MAX_SEED,
        var_id: MAX_SEED,
        seed3: MAX_SEED,
        seed4: MAX_SEED,
    };
    for seeds in [HdrSeeds::default(), max] {
        for counter in [0, 1, MAX_SEED - 1, MAX_SEED] {
            let u = hdr_uniform(counter, &seeds).unwrap();
            assert!(u > 0.0 && u < 1.0, "{u}");
        }
    }
    let mut tail = [0.0; 2];
    hdr_fill(&max, MAX_SEED - 1, &mut tail).unwrap();
    assert_eq!(tail[1], hdr_uniform(MAX_SEED, &max).unwrap());
}

#[test]
fn rejects_out_of_range_inputs() {
    let over = MAX_SEED + 1;
    let err = |name, value| Err(HdrError::SeedOutOfRange { name, value });
    let ok = HdrSeeds::default();
    assert_eq!(hdr_uniform(over, &ok), err("counter", over.into()));
    assert_eq!(
        hdr_uniform(1, &HdrSeeds { entity: over, ..ok }),
        err("entity", over.into())
    );
    assert_eq!(
        hdr_uniform(1, &HdrSeeds { var_id: over, ..ok }),
        err("var_id", over.into())
    );
    assert_eq!(
        hdr_uniform(1, &HdrSeeds { seed3: over, ..ok }),
        err("seed3", over.into())
    );
    assert_eq!(
        hdr_uniform(
            1,
            &HdrSeeds {
                seed4: u32::MAX,
                ..ok
            }
        ),
        err("seed4", u32::MAX.into())
    );
}

#[test]
fn fill_rejects_run_past_max_without_writing() {
    let mut out = [-1.0; 3];
    assert_eq!(
        hdr_fill(&HdrSeeds::default(), MAX_SEED - 1, &mut out),
        Err(HdrError::SeedOutOfRange {
            name: "counter",
            value: u64::from(MAX_SEED) + 1
        })
    );
    assert_eq!(out, [-1.0; 3]);
    assert!(hdr_fill(&HdrSeeds::default(), MAX_SEED + 1, &mut []).is_err());
}
