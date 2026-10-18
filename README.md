# sipkit

A kernel for the [SIPmath 3.1 Standard](https://www.probabilitymanagement.org/s/SIPmath-31-Standard-2025-01-31.pdf) in three languages:

| Path | Language | Role |
| :--- | :--- | :--- |
| `crates/sipkit` | Rust | Core kernel: HDR 2.0 PRNG, Metalog quantile |
| `crates/sipkit-wasm` | Rust → WASM | `wasm-bindgen` exports of the kernel |
| `packages/wasm` | JS (generated) | `@sipkit/wasm`: built from `crates/sipkit-wasm` into `pkg/` |
| `packages/core` | TypeScript | `@sipkit/core`: pure-TS HDR + Metalog for light simulation |
| `lean` | Lean 4 + Mathlib | Proofs: logit derivative/monotonicity, 2-term Metalog monotonicity |
| `fixtures` | JSON | Shared test vectors every implementation must reproduce |

## Toolchain

- Rust 1.95.0 with `wasm32-unknown-unknown` (pinned in `rust-toolchain.toml`)
- `wasm-bindgen-cli` at the exact version pinned in `crates/sipkit-wasm/Cargo.toml`
- Node ≥ 22, pnpm 11
- elan; Lean toolchain pinned in `lean/lean-toolchain` (fetch Mathlib build cache with `cd lean && lake exe cache get`)

## Commands

```sh
pnpm install
pnpm test          # cargo test --workspace, pnpm -r test (builds wasm first), lake build
pnpm run lint      # rustfmt, clippy -D warnings, tsc --noEmit
pnpm run build     # @sipkit/core dist/ and @sipkit/wasm pkg/
```

## Fixtures

- `fixtures/hdr-2.0-seed-standard.json`: 10,000 HDR outputs (counter 1–10000, entity 1, varId 2, seed3 3, seed4 4) extracted from ProbabilityManagement.org's [HDR_2_0-Seed-Standard.xlsx](https://www.probabilitymanagement.org/s/HDR_2_0-Seed-Standard.xlsx). Every implementation must match them exactly.
- `fixtures/metalog-quantile.json`: Metalog quantiles for k ∈ {2…10, 12, 16} × all four bound types; tolerance 1e-12 relative.

## Design notes

- **HDR.** The formula is the HDR 2.0 standard from the seed-standard workbook (five seeds: counter, entity, varId, seed3, seed4). Seeds are restricted to 8 decimal digits (`MAX_SEED = 99_999_999`); within that domain every intermediate stays below 2^53, so TypeScript uses plain `number` arithmetic, not `BigInt`.
- **Metalog domain.** `quantile(y)` returns `NaN` outside the open interval (0, 1) instead of clamping. Bounded form is evaluated as `bL + (bU − bL) / (1 + e^−M)` to avoid `inf/inf` overflow.
- **Lean.** A 2-term Metalog is strictly increasing on (0, 1) if and only if `a₂ > 0`; both directions are proved.

## Not yet implemented

Components of the Standard without code here, each tracked as an issue: table lookup ([#1](https://github.com/gavmor/sipkit/issues/1)), Cholesky / Gaussian copula ([#2](https://github.com/gavmor/sipkit/issues/2)), SIPmath JSON schema validation ([#3](https://github.com/gavmor/sipkit/issues/3)), C FFI library ([#4](https://github.com/gavmor/sipkit/issues/4)), `rayon` multithreading ([#5](https://github.com/gavmor/sipkit/issues/5)).

Packages are distributed through GitHub only, not npm or crates.io.

## References

- D. W. Hubbard, "A Multi-Dimensional, Counter-Based Pseudo Random Number Generator as a Standard for Monte Carlo Simulations," *2019 Winter Simulation Conference*, [doi:10.1109/WSC40007.2019.9004773](https://doi.org/10.1109/WSC40007.2019.9004773).
- T. W. Keelin, "The Metalog Distributions," *Decision Analysis* 13(4), 2016, [doi:10.1287/deca.2016.0338](https://doi.org/10.1287/deca.2016.0338).
- ProbabilityManagement.org, [SIPmath 3.1 Standard](https://www.probabilitymanagement.org/s/SIPmath-31-Standard-2025-01-31.pdf) and [HDR 2.0 seed standard workbook](https://www.probabilitymanagement.org/s/HDR_2_0-Seed-Standard.xlsx) (source of the HDR fixture).

## License

[MIT](LICENSE).

sipkit is an independent implementation. It is not affiliated with or endorsed by ProbabilityManagement.org. SIPmath™ is a trademark of ProbabilityManagement.org.
