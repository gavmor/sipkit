//! Hubbard Decision Research HDR 2.0 counter-based PRNG (ProbabilityManagement.org seed standard).
//!
//! Every input is an integer in `[0, MAX_SEED]`. Within that domain every intermediate product
//! is below 2^53, so `u64` arithmetic never overflows and results match the reference
//! spreadsheet bit for bit.

use std::fmt;

/// Largest admissible counter or seed value (8 decimal digits).
pub const MAX_SEED: u32 = 99_999_999;

/// Seeds identifying an independent HDR stream. `seed3`/`seed4` default to 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HdrSeeds {
    pub entity: u32,
    pub var_id: u32,
    pub seed3: u32,
    pub seed4: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrError {
    /// `name` is the offending input (`counter`, `entity`, `var_id`, `seed3`, `seed4`).
    SeedOutOfRange { name: &'static str, value: u64 },
}

impl fmt::Display for HdrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SeedOutOfRange { name, value } => {
                write!(f, "HDR {name} = {value} is outside [0, {MAX_SEED}]")
            }
        }
    }
}

impl std::error::Error for HdrError {}

const K: u64 = 999_999_999_999_989;
const TWO_32: f64 = 4_294_967_296.0;

fn check(name: &'static str, value: u64) -> Result<(), HdrError> {
    if value > u64::from(MAX_SEED) {
        Err(HdrError::SeedOutOfRange { name, value })
    } else {
        Ok(())
    }
}

fn check_seeds(seeds: &HdrSeeds) -> Result<(), HdrError> {
    check("entity", seeds.entity.into())?;
    check("var_id", seeds.var_id.into())?;
    check("seed3", seeds.seed3.into())?;
    check("seed4", seeds.seed4.into())
}

/// Stream-constant parts of the two linear combinations, hoisted out of per-trial work.
#[derive(Clone, Copy)]
struct Offsets {
    o1: u64,
    o2: u64,
}

impl Offsets {
    fn new(s: &HdrSeeds) -> Self {
        let (v, e, s3, s4) = (
            u64::from(s.var_id),
            u64::from(s.entity),
            u64::from(s.seed3),
            u64::from(s.seed4),
        );
        Self {
            o1: v * 1_800_451 + e * 2_000_371 + s3 * 1_796_777 + s4 * 2_299_603,
            o2: v * 2_399_993 + e * 2_100_869 + s3 * 1_918_303 + s4 * 1_624_729,
        }
    }

    /// Requires `counter <= MAX_SEED` and offsets built from in-range seeds.
    #[inline]
    fn uniform(self, counter: u64) -> f64 {
        let x1 = (counter * 2_499_997 + self.o1) % 7_450_589;
        let t1 = ((K % (x1 * 4_658 + 7_450_581)) * 383) % 99_991;
        let x2 = (counter * 2_246_527 + self.o2) % 7_450_987;
        let t2 = ((K % (x2 * 7_580 + 7_560_584)) * 17_669) % 7_440_893;
        let bits = ((t1 * 7_440_893 + t2) * 1_343) % (1 << 32);
        (bits as f64 + 0.5) / TWO_32
    }
}

/// HDR 2.0 uniform in (0, 1) for one trial `counter`.
pub fn hdr_uniform(counter: u32, seeds: &HdrSeeds) -> Result<f64, HdrError> {
    check("counter", counter.into())?;
    check_seeds(seeds)?;
    Ok(Offsets::new(seeds).uniform(counter.into()))
}

/// `out[i] = HDR(first_counter + i)`; validates seeds and the last counter once, then runs an
/// unchecked loop. Leaves `out` untouched on error.
pub fn hdr_fill(seeds: &HdrSeeds, first_counter: u32, out: &mut [f64]) -> Result<(), HdrError> {
    check_seeds(seeds)?;
    check("counter", first_counter.into())?;
    if let Some(span) = out.len().checked_sub(1) {
        check(
            "counter",
            u64::from(first_counter).saturating_add(span as u64),
        )?;
    }
    let offsets = Offsets::new(seeds);
    for (counter, slot) in (u64::from(first_counter)..).zip(out.iter_mut()) {
        *slot = offsets.uniform(counter);
    }
    Ok(())
}
