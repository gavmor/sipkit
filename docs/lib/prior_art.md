Several existing open-source libraries, packages, and reference implementations can be leveraged or audited as dependencies across the Rust, TypeScript, Python, and R ecosystems.

---

### 1. Metalog Distributions

| Ecosystem | Package / Repository | Status & Suitability | Notes & Caveats |
| --- | --- | --- | --- |
| **TypeScript / JS** | `@quri/metalog` (npm)

 | **Candidate Dependency** | Developed by the Quantified Uncertainty Research Institute (QURI) for the Squiggle probabilistic language. Implements quantile evaluations and fits.

 |
| **TypeScript / JS** | `metalog` (npm)

 | **Do Not Use (Name Collision)** | This package is an asynchronous server logger from Metarhia, completely unrelated to probability distributions.

 |
| **Rust** | `metalog` (crates.io `v0.1.0` by misalcedo)

 | **Early Stage / Experimental** | Implements basic Metalog distribution generation and a CLI. Suitable for inspection, but for production kernels, embedding the 16-term basis expansion directly avoids third-party churn.

 |
| **Python** | `pymetalog` (PyPI / GitHub: `tjefferies/pymetalog`)

 | **Reference Baseline** | The primary Python port of Keelin’s work. Excellent for generating golden test fixtures to validate your Rust/TypeScript outputs.

 |
| **Python** | `metalogistic` (PyPI / GitHub: `portchain/metalogistic`)

 | **Reference Baseline** | Integrates Metalog distributions into the `scipy.stats.rv_continuous` framework.

 |
| **R** | `rmetalog` (CRAN)

 | **Authoritative Reference** | Maintained by Isaac Faber alongside Tom Keelin. Serves as the gold standard for verifying Metalog 1.0 fitting and quantile evaluation logic.

 |

---

### 2. Hubbard HDR 2.0 Counter PRNG

* **Current Package Status:** There are **no official standalone npm packages or crates.io packages** specifically published for Doug Hubbard’s HDR PRNG.


* **Reference Implementations:**
* **R:** The `probability-management/R-SIPmath-Tools` repository contains the reference R routines for generating HDR streams.


* **Excel / VBA:** The standard seed workbook `HDR_2_0-Seed-Standard.xlsx` published by ProbabilityManagement.org contains the canonical single-cell Excel formula.




* **Dependency Recommendation:** **Vendor directly (zero external dependencies).** Because Hubbard HDR 2.0 consists of roughly 10 lines of 64-bit integer bit-shifting, XOR scrambling, and modular arithmetic, pulling in an unverified third-party micro-package is an unnecessary supply-chain risk. In TypeScript, implementing it with native `BigInt` guarantees standard 64-bit precision.



---

### 3. SIPmath 3.1 JSON Schema Validation & Interchange

| Ecosystem | Recommended Dependencies | Implementation Role |
| --- | --- | --- |
| **TypeScript** | `zod` or `ajv` (JSON Schema Draft-07) | Directly ingests and validates `multi-lib-schema-2021-08-19.json`. `zod` allows you to infer static TypeScript types from the schema definitions.

 |
| **Rust** | `serde` and `serde_json` | The industry-standard choice for strongly typed, zero-copy deserialization of `.SIPmath` JSON files into Rust structs. |
| **Python** | `PySIPmath` (`probability-management/PySIPmath`)

 | Official tool from ProbabilityManagement.org that converts pandas DataFrames into SIPmath 3.0/3.1 JSON files and `.xlsx` workbooks.

 |

---

### 4. Gaussian Copulas and Linear Algebra

To evaluate multivariate Gaussian copulas (Canonical Library 02), the runtime requires standard normal cumulative distribution ($\Phi$) and quantile ($\Phi^{-1}$ / probit) functions, along with lower-triangular Cholesky factor multiplication:

* **Rust Dependencies:**
* `statrs`: Provides high-precision standard normal CDF ($\Phi$) and inverse CDF / quantile ($\Phi^{-1}$).
* `faer` or `nalgebra`: High-performance, pure-Rust linear algebra crates for computing Cholesky decompositions ($\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T$) and matrix-vector products.


* **TypeScript / Browser Dependencies:**
* `simple-statistics` or `jstat` (npm): Provides Beasley-Springer-Moro or Acklam rational approximations for standard normal CDF and inverse CDF.
* `ml-matrix` (npm): Lightweight linear algebra library for matrix operations and Cholesky decomposition.



Would you like me to do a more in-depth research instead and propose a research plan?
