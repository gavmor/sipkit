# sipmath

SIPmath kernel in three languages, per `sipmath_implementation_blueprint.md`:

| Path | Language | Role |
| :--- | :--- | :--- |
| `crates/sipmath` | Rust | Core kernel: HDR 2.0 PRNG, Metalog quantile |
| `crates/sipmath-wasm` | Rust → WASM | `wasm-bindgen` exports of the kernel |
| `packages/wasm` | JS (generated) | `@sipmath/wasm`: built from `crates/sipmath-wasm` into `pkg/` |
| `packages/core` | TypeScript | `@sipmath/core`: pure-TS HDR + Metalog for light simulation |
| `lean` | Lean 4 + Mathlib | Proofs: logit derivative/monotonicity, 2-term Metalog monotonicity |
| `fixtures` | JSON | Shared test vectors every implementation must reproduce |

## Toolchain

- Rust 1.95.0 with `wasm32-unknown-unknown` (pinned in `rust-toolchain.toml`)
- `wasm-bindgen-cli` at the exact version pinned in `crates/sipmath-wasm/Cargo.toml`
- Node ≥ 22, pnpm 11
- elan; Lean toolchain pinned in `lean/lean-toolchain` (fetch Mathlib build cache with `cd lean && lake exe cache get`)

## Commands

```sh
pnpm install
pnpm test          # cargo test --workspace, pnpm -r test (builds wasm first), lake build
pnpm run lint      # rustfmt, clippy -D warnings, tsc --noEmit
pnpm run build     # @sipmath/core dist/ and @sipmath/wasm pkg/
```

## Fixtures

- `fixtures/hdr-2.0-seed-standard.json`: 10,000 HDR outputs (counter 1–10000, entity 1, varId 2, seed3 3, seed4 4) extracted from ProbabilityManagement.org's [HDR_2_0-Seed-Standard.xlsx](https://www.probabilitymanagement.org/s/HDR_2_0-Seed-Standard.xlsx). Every implementation must match them exactly.
- `fixtures/metalog-quantile.json`: Metalog quantiles for k ∈ {2…10, 12, 16} × all four bound types; tolerance 1e-12 relative.

## Deviations from the blueprint

- **HDR.** The blueprint's `hdr_uniform` is not Hubbard's generator. The implemented formula is the HDR 2.0 standard from the seed-standard workbook (five seeds: counter, entity, varId, seed3, seed4). Seeds are restricted to 8 decimal digits (`MAX_SEED = 99_999_999`); within that domain every intermediate stays below 2^53, so TypeScript uses plain `number` arithmetic, not `BigInt`.
- **Metalog domain.** `quantile(y)` returns `NaN` outside the open interval (0, 1) instead of clamping. Bounded form is evaluated as `bL + (bU − bL) / (1 + e^−M)` to avoid `inf/inf` overflow.
- **Lean.** The blueprint's 2-term theorem claims an iff but states one direction; both directions are proved.

## Not yet implemented

Blueprint components without code here: table lookup (empirical inverse CDF), Cholesky / Gaussian copula, SIPmath JSON schema validation (Zod), C FFI library, `rayon` multithreading.
