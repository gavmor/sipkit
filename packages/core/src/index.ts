/**
 * @gavmor/sipkit-core — Thin TypeScript orchestration & validation shell.
 *
 * Contains ZERO probability math: all HDR 2.0, Metalog, Probit, Normal CDF,
 * Cholesky, Lookup, and Welford reductions execute in the Rust WASM kernel
 * (`@gavmor/sipkit-wasm`).
 *
 * Responsibilities:
 * 1. Zod validation of SIPmath 3.1 JSON libraries (Canonical Libs 01–10).
 * 2. Resolution of user parameters (Lib 09), generalized Metalog tables (Lib 10),
 *    and sparse upper-triangular correlation matrices (Lib 02).
 * 3. External array loading hooks for CSV/XLSX SIP_Array references (Libs 03–05).
 * 4. Batch orchestration over `@gavmor/sipkit-wasm` returning `Float64Array`s.
 */

import { z } from "zod";

// ---------------------------------------------------------------------------
// 1. WASM Kernel Contract (matches wasm-bindgen output of `crates/sipkit-wasm`)
// ---------------------------------------------------------------------------

export const MAX_SEED = 99_999_999;

export const WasmRounding = {
  None: 0,
  Nearest: 1,
  Up: 2,
  Down: 3,
} as const;

export type WasmRoundingCode = (typeof WasmRounding)[keyof typeof WasmRounding];

export interface TrialStats {
  readonly count: number;
  readonly nanCount: number;
  readonly mean: number;
  readonly variance: number;
  readonly stdDev: number;
  readonly min: number;
  readonly max: number;
}

export interface WasmTrialStatsHandle extends TrialStats {
  merge(other: WasmTrialStatsHandle): WasmTrialStatsHandle;
  free(): void;
}

export interface SipkitWasmKernel {
  hdrBatch(
    startCounter: number,
    count: number,
    entity: number,
    varId: number,
    seed3: number,
    seed4: number
  ): Float64Array;

  metalogBatch(
    uniforms: Float64Array,
    aCoeffs: Float64Array,
    lowerBound: number | undefined,
    upperBound: number | undefined,
    rounding: WasmRoundingCode
  ): Float64Array;

  hdrMetalogBatch(
    startCounter: number,
    count: number,
    entity: number,
    varId: number,
    seed3: number,
    seed4: number,
    aCoeffs: Float64Array,
    lowerBound: number | undefined,
    upperBound: number | undefined,
    rounding: WasmRoundingCode
  ): Float64Array;

  hdrMetalogStats(
    startCounter: number,
    count: number,
    entity: number,
    varId: number,
    seed3: number,
    seed4: number,
    aCoeffs: Float64Array,
    lowerBound: number | undefined,
    upperBound: number | undefined,
    rounding: WasmRoundingCode
  ): WasmTrialStatsHandle;

  lookupBatch(
    uniforms: Float64Array,
    cutoffs: Float64Array,
    values: Float64Array
  ): Float64Array;

  hdrLookupBatch(
    startCounter: number,
    count: number,
    entity: number,
    varId: number,
    seed3: number,
    seed4: number,
    cutoffs: Float64Array,
    values: Float64Array
  ): Float64Array;

  choleskyDecompose(dim: number, corrFlat: Float64Array): Float64Array;

  gaussianCopulaBatch(
    dim: number,
    startCounter: number,
    count: number,
    seedsFlat: Uint32Array,
    choleskyL: Float64Array
  ): Float64Array;

  summarizeTrials(trials: Float64Array): WasmTrialStatsHandle;

  normalQuantile(p: number): number;

  normalCdf(x: number): number;
}

/**
 * Helper to copy a WASM stats handle into a plain JS object and immediately
 * release the underlying WASM heap allocation.
 */
function materializeStats(handle: WasmTrialStatsHandle): TrialStats {
  try {
    return {
      count: handle.count,
      nanCount: handle.nanCount,
      mean: handle.mean,
      variance: handle.variance,
      stdDev: handle.stdDev,
      min: handle.min,
      max: handle.max,
    };
  } finally {
    handle.free();
  }
}

// ---------------------------------------------------------------------------
// 2. SIPmath 3.1 Zod Validation Schemas (Canonical Libs 01–10)
// ---------------------------------------------------------------------------

export const ParameterRefSchema = z.object({
  type: z.literal("parameter"),
  name: z.string().min(1),
});

export const GlobalVarRefSchema = z.object({
  type: z.literal("globalVariables"),
  value: z.string().min(1),
});

export const SeedValueSchema = z.union([
  z.number().int().min(0).max(MAX_SEED),
  ParameterRefSchema,
]);

export const HdrRngNodeSchema = z.object({
  name: z.string().min(1),
  function: z.literal("HDR_2_0"),
  arguments: z.object({
    counter: z.union([z.literal("PM_Index"), z.number().int().min(0)]),
    entity: SeedValueSchema.default(0),
    varId: SeedValueSchema.default(0),
    seed3: SeedValueSchema.default(0),
    seed4: SeedValueSchema.default(0),
  }),
});

export const IndexRngNodeSchema = z.object({
  name: z.string().min(1),
  function: z.literal("Index"),
  arguments: z.object({
    counter: z.literal("PM_Index"),
  }),
});

export const RngNodeSchema = z.discriminatedUnion("function", [
  HdrRngNodeSchema,
  IndexRngNodeSchema,
]);

/** Sparse upper-triangular correlation matrix format used in Canonical Lib 02. */
export const CorrelationMatrixValueSchema = z.object({
  columns: z.array(z.string().min(1)).min(1),
  rows: z.array(z.string().min(1)).min(1),
  matrix: z.array(
    z.object({
      row: z.string().min(1),
      col: z.string().min(1),
      value: z.number().min(-1).max(1),
    })
  ),
});

/** Generalized Metalog row format used in Canonical Lib 10. */
export const GeneralizedMetalogRowSchema = z
  .object({
    aCoefficients: z.array(z.number()).min(2),
  })
  .catchall(z.union([z.number(), z.array(z.number())]));

export const GaussianCopulaNodeSchema = z.object({
  name: z.string().min(1),
  function: z.literal("GaussianCopula"),
  arguments: z.object({
    correlationMatrix: GlobalVarRefSchema,
    rng: z.array(z.string().min(1)).min(1),
  }),
  copulaLayer: z.array(z.string().min(1)).min(1),
});

export const ArrayCopulaNodeSchema = z.object({
  name: z.string().min(1),
  function: z.literal("SIP_Array"),
  arguments: z.object({
    type: z.enum(["csv", "xlsx", "array"]),
    url: z.string().url().optional(),
    value: z.array(z.string().min(1)).min(1),
  }),
  copulaLayer: z.array(z.string().min(1)).min(1),
});

export const CopulaNodeSchema = z.discriminatedUnion("function", [
  GaussianCopulaNodeSchema,
  ArrayCopulaNodeSchema,
]);

export const SipRefSchema = z.object({
  source: z.enum(["rng", "copula"]),
  name: z.string().min(1),
  copulaLayer: z.string().min(1).optional(),
});

export const RoundingSchema = z.enum(["nearest", "up", "down"]);

export const MetalogSipSchema = z.object({
  name: z.string().min(1),
  environment: z.string().optional(),
  ref: SipRefSchema,
  function: z.enum(["Metalog_1_0", "Metalog_2_0"]),
  rounding: RoundingSchema.optional(),
  arguments: z.object({
    aCoefficients: z.array(z.number()).min(2),
    lowerBound: z.number().optional(),
    upperBound: z.number().optional(),
  }),
  metadata: z.record(z.unknown()).optional(),
});

export const GeneralizedMetalogSipSchema = z.object({
  name: z.string().min(1),
  environment: z.string().optional(),
  ref: SipRefSchema,
  function: z.literal("GeneralizedMetalog"),
  rounding: RoundingSchema.optional(),
  arguments: z.object({
    aCoefficients: GlobalVarRefSchema,
    dims: z.array(
      z.object({
        type: z.literal("parameter"),
        name: z.string().min(1),
        numericType: z.enum(["integer", "float"]),
      })
    ),
    lowerBound: z.number().optional(),
    upperBound: z.number().optional(),
  }),
  metadata: z.record(z.unknown()).optional(),
});

export const LookupTableSipSchema = z.object({
  name: z.string().min(1),
  environment: z.string().optional(),
  ref: SipRefSchema,
  function: z.literal("Lookup_Table"),
  arguments: z.object({
    value: z.array(z.tuple([z.number(), z.number()])).min(1),
  }),
  metadata: z.record(z.unknown()).optional(),
});

export const SipArraySipSchema = z.object({
  name: z.string().min(1),
  environment: z.string().optional(),
  ref: SipRefSchema,
  function: z.literal("SIP_Array"),
  arguments: z.object({
    type: z.enum(["array", "csv", "xlsx"]),
    url: z.string().url().optional(),
    value: z.union([z.array(z.number()), z.string().min(1)]),
  }),
  metadata: z.record(z.unknown()).optional(),
});

/** Environment-specific functions like PsiNormal / RiskNormal (Canonical Lib 09). */
export const EnvironmentSpecificSipSchema = z.object({
  name: z.string().min(1),
  environment: z.string().min(1),
  ref: SipRefSchema,
  function: z.string().min(1),
  arguments: z.record(z.unknown()),
  metadata: z.record(z.unknown()).optional(),
});

export const SipNodeSchema = z.union([
  MetalogSipSchema,
  GeneralizedMetalogSipSchema,
  LookupTableSipSchema,
  SipArraySipSchema,
  EnvironmentSpecificSipSchema,
]);

export const SipMathModelSchema = z.object({
  name: z.string().min(1),
  // Optional to accept Canonical Lib 01 (which omits objectType/libraryType);
  // enforced when `parseSipModel(raw, { strict: true })` is called.
  objectType: z.literal("sipModel").optional(),
  libraryType: z.string().startsWith("SIPmath").optional(),
  dateCreated: z.string().optional(),
  version: z.union([z.string(), z.number()]).optional(),
  provenance: z.string().optional(),
  PM_Trials: z.number().int().positive().optional(),
  comments: z.array(z.record(z.unknown())).optional(),
  globalVariables: z
    .array(
      z.object({
        name: z.string().min(1),
        value: z.unknown(),
      })
    )
    .optional(),
  U01: z.object({
    rng: z.array(RngNodeSchema).optional(),
    copula: z.array(CopulaNodeSchema).optional(),
  }),
  sips: z.array(SipNodeSchema).min(1),
});

export type SipMathModel = z.infer<typeof SipMathModelSchema>;
export type SipNode = z.infer<typeof SipNodeSchema>;

/**
 * Parses and validates an untrusted JSON payload against the SIPmath 3.1 schema.
 */
export function parseSipModel(
  raw: unknown,
  options: { strict?: boolean | undefined } = {}
): SipMathModel {
  const model = SipMathModelSchema.parse(raw);
  if (options.strict && model.objectType !== "sipModel") {
    throw new Error('Strict SIPmath validation failed: missing top-level objectType="sipModel"');
  }
  return model;
}

// ---------------------------------------------------------------------------
// 3. External Array Loader Hook (Libs 03, 04, 05)
// ---------------------------------------------------------------------------

export interface ExternalArrayRequest {
  type: "csv" | "xlsx";
  url: string;
  columnOrRange: string;
}

export type ExternalArrayLoader = (
  req: ExternalArrayRequest
) => Promise<Float64Array>;

// ---------------------------------------------------------------------------
// 4. Orchestration Runner (`SipModelRunner`)
// ---------------------------------------------------------------------------

export interface EvaluateOptions {
  /** Number of Monte Carlo trials to generate. */
  count: number;
  /** Starting `PM_Index` counter (defaults to `1`, matching SIPmath 1-based trials). */
  startCounter?: number;
  /** Runtime values for `{ type: "parameter", name }` references (Libs 09 & 10). */
  parameters?: Record<string, number>;
}

function toWasmRounding(
  mode?: z.infer<typeof RoundingSchema>
): WasmRoundingCode {
  switch (mode) {
    case "nearest":
      return WasmRounding.Nearest;
    case "up":
      return WasmRounding.Up;
    case "down":
      return WasmRounding.Down;
    default:
      return WasmRounding.None;
  }
}

/**
 * Expands the sparse upper-triangular SIPmath correlation matrix format (Lib 02)
 * into a dense symmetric row-major `Float64Array` ready for `wasm.choleskyDecompose`.
 */
function buildDenseCorrelationMatrix(rawValue: unknown): {
  dim: number;
  flat: Float64Array;
} {
  const parsed = CorrelationMatrixValueSchema.parse(rawValue);
  const dim = parsed.rows.length;
  if (parsed.columns.length !== dim) {
    throw new Error("Correlation matrix must be square (rows.length !== columns.length)");
  }

  const rowIndex = new Map(parsed.rows.map((name, idx) => [name, idx]));
  const colIndex = new Map(parsed.columns.map((name, idx) => [name, idx]));

  // Start with identity matrix so omitted diagonal entries default to 1.0
  const flat = new Float64Array(dim * dim);
  for (let i = 0; i < dim; i++) {
    flat[i * dim + i] = 1.0;
  }

  for (const entry of parsed.matrix) {
    const r = rowIndex.get(entry.row);
    const c = colIndex.get(entry.col);
    if (r === undefined || c === undefined) {
      throw new Error(
        `Correlation matrix references unknown row/col ("${entry.row}", "${entry.col}")`
      );
    }
    // Mirror across diagonal so upper-triangular specs become symmetric
    flat[r * dim + c] = entry.value;
    flat[c * dim + r] = entry.value;
  }

  return { dim, flat };
}

/**
 * High-level runner for a validated SIPmath 3.1 library backed by `@gavmor/sipkit-wasm`.
 */
export class SipModelRunner {
  private readonly wasm: SipkitWasmKernel;
  private readonly model: SipMathModel;
  private readonly externalLoader: ExternalArrayLoader | undefined;

  private readonly globalVars = new Map<string, unknown>();
  private readonly rngMap = new Map<string, z.infer<typeof RngNodeSchema>>();
  private readonly copulaMap = new Map<string, z.infer<typeof CopulaNodeSchema>>();
  private readonly choleskyCache = new Map<string, { dim: number; L: Float64Array }>();
  private readonly portableSips = new Map<string, SipNode>();

  constructor(
    wasm: SipkitWasmKernel,
    rawModel: unknown,
    options: { strict?: boolean | undefined; externalLoader?: ExternalArrayLoader | undefined } = {}
  ) {
    this.wasm = wasm;
    this.model = parseSipModel(rawModel, { strict: options.strict });
    this.externalLoader = options.externalLoader;

    for (const gv of this.model.globalVariables ?? []) {
      this.globalVars.set(gv.name, gv.value);
    }
    for (const rng of this.model.U01.rng ?? []) {
      this.rngMap.set(rng.name, rng);
    }
    for (const copula of this.model.U01.copula ?? []) {
      this.copulaMap.set(copula.name, copula);
      if (copula.function === "GaussianCopula") {
        const rawMatrix = this.globalVars.get(
          copula.arguments.correlationMatrix.value
        );
        if (rawMatrix === undefined) {
          throw new Error(
            `Missing globalVariable "${copula.arguments.correlationMatrix.value}" for copula "${copula.name}"`
          );
        }
        const { dim, flat } = buildDenseCorrelationMatrix(rawMatrix);
        if (dim !== copula.arguments.rng.length || dim !== copula.copulaLayer.length) {
          throw new Error(
            `Copula "${copula.name}" dimension mismatch: matrix is ${dim}x${dim}, rng has ${copula.arguments.rng.length}, copulaLayer has ${copula.copulaLayer.length}`
          );
        }
        // Precompute and validate Cholesky factor once in Rust/WASM
        const L = this.wasm.choleskyDecompose(dim, flat);
        this.choleskyCache.set(copula.name, { dim, L });
      } else if (copula.arguments.value.length !== copula.copulaLayer.length) {
        throw new Error(
          `Array copula "${copula.name}" column mismatch: value has ${copula.arguments.value.length}, copulaLayer has ${copula.copulaLayer.length}`
        );
      }
    }

    // Filter out environment-specific overrides (e.g., ExcelFrontline/ExcelPalisade in Lib 09)
    // and retain portable definitions keyed by SIP name.
    for (const sip of this.model.sips) {
      if (!sip.environment) {
        this.portableSips.set(sip.name, sip);
      }
    }
  }

  /** Lists all portable SIP variable names available in this model. */
  public get sipNames(): string[] {
    return Array.from(this.portableSips.keys());
  }

  private resolveSeed(
    seed: z.infer<typeof SeedValueSchema>,
    params: Record<string, number>
  ): number {
    const resolved =
      typeof seed === "number" ? seed : params[seed.name];
    if (resolved === undefined) {
      const paramName = typeof seed === "number" ? "" : seed.name;
      throw new Error(`Missing required runtime parameter "${paramName}"`);
    }
    if (!Number.isInteger(resolved) || resolved < 0 || resolved > MAX_SEED) {
      throw new RangeError(
        `Resolved HDR seed (${resolved}) must be an integer in [0, ${MAX_SEED}]`
      );
    }
    return resolved;
  }

  private resolveHdrSeeds(
    rngName: string,
    params: Record<string, number>
  ): [entity: number, varId: number, seed3: number, seed4: number] {
    const rng = this.rngMap.get(rngName);
    if (!rng) {
      throw new Error(`Unknown RNG reference "${rngName}"`);
    }
    if (rng.function !== "HDR_2_0") {
      throw new Error(`Expected HDR_2_0 RNG for "${rngName}", found "${rng.function}"`);
    }
    return [
      this.resolveSeed(rng.arguments.entity, params),
      this.resolveSeed(rng.arguments.varId, params),
      this.resolveSeed(rng.arguments.seed3, params),
      this.resolveSeed(rng.arguments.seed4, params),
    ];
  }

  private resolveGeneralizedCoefficients(
    sip: z.infer<typeof GeneralizedMetalogSipSchema>,
    params: Record<string, number>
  ): Float64Array {
    const rawTable = this.globalVars.get(sip.arguments.aCoefficients.value);
    if (!Array.isArray(rawTable)) {
      throw new Error(
        `Missing globalVariables table "${sip.arguments.aCoefficients.value}" for GeneralizedMetalog "${sip.name}"`
      );
    }

    const rows = z.array(GeneralizedMetalogRowSchema).parse(rawTable);
    const match = rows.find((row) =>
      sip.arguments.dims.every((dim) => {
        const expected = params[dim.name];
        if (expected === undefined) {
          throw new Error(
            `Missing required GeneralizedMetalog parameter "${dim.name}"`
          );
        }
        // Tolerate minor casing inconsistencies found in Canonical Lib 10 PDF ("numberofIIDs" vs "numberOfIIDs")
        const actualKey = Object.keys(row).find(
          (k) => k.toLowerCase() === dim.name.toLowerCase()
        );
        const actual = actualKey !== undefined ? row[actualKey] : undefined;
        if (typeof actual !== "number") return false;
        return dim.numericType === "integer"
          ? Math.round(actual) === Math.round(expected)
          : Math.abs(actual - expected) < 1e-6;
      })
    );

    if (!match) {
      throw new Error(
        `No matching GeneralizedMetalog coefficient row for "${sip.name}" with parameters ${JSON.stringify(params)}`
      );
    }
    return Float64Array.from(match.aCoefficients);
  }

  /**
   * Evaluates all copulas referenced by the requested SIPs once per batch,
   * returning zero-copy `Float64Array` layer views keyed by `${copulaName}:${layerName}`.
   */
  private async evaluateCopulaLayers(
    copulaNames: Set<string>,
    startCounter: number,
    count: number,
    params: Record<string, number>
  ): Promise<Map<string, Float64Array>> {
    const layerMap = new Map<string, Float64Array>();

    for (const copulaName of copulaNames) {
      const copula = this.copulaMap.get(copulaName);
      if (!copula) {
        throw new Error(`Unknown copula reference "${copulaName}"`);
      }

      if (copula.function === "GaussianCopula") {
        const cached = this.choleskyCache.get(copulaName);
        if (!cached) {
          throw new Error(`Cholesky factor missing for copula "${copulaName}"`);
        }

        const packedSeeds = new Uint32Array(cached.dim * 4);
        copula.arguments.rng.forEach((rngName, d) => {
          const [e, v, s3, s4] = this.resolveHdrSeeds(rngName, params);
          packedSeeds[d * 4] = e;
          packedSeeds[d * 4 + 1] = v;
          packedSeeds[d * 4 + 2] = s3;
          packedSeeds[d * 4 + 3] = s4;
        });

        // Single WASM call produces column-major buffer of length dim * count
        const colMajor = this.wasm.gaussianCopulaBatch(
          cached.dim,
          startCounter,
          count,
          packedSeeds,
          cached.L
        );

        for (let d = 0; d < cached.dim; d++) {
          const layerView = colMajor.subarray(d * count, (d + 1) * count);
          layerMap.set(`${copulaName}:${copula.copulaLayer[d]}`, layerView);
        }
      } else {
        // Empirical array copula (Canonical Lib 05: Happy Fish)
        if (!this.externalLoader || !copula.arguments.url) {
          throw new Error(
            `ExternalArrayLoader and URL are required to evaluate array copula "${copulaName}"`
          );
        }
        const extType = copula.arguments.type;
        if (extType === "array") {
          throw new Error(`Inline array copula not supported for "${copulaName}"`);
        }
        // One column/range per layer (lengths checked in the constructor).
        for (const [d, columnOrRange] of copula.arguments.value.entries()) {
          const col = await this.externalLoader({
            type: extType,
            url: copula.arguments.url,
            columnOrRange,
          });
          const slice = col.subarray(startCounter - 1, startCounter - 1 + count);
          layerMap.set(`${copulaName}:${copula.copulaLayer[d]}`, slice);
        }
      }
    }

    return layerMap;
  }

  private async evaluateSingleSip(
    sip: SipNode,
    startCounter: number,
    count: number,
    params: Record<string, number>,
    copulaLayers: Map<string, Float64Array>
  ): Promise<Float64Array> {
    if (sip.function === "SIP_Array") {
      const args = sip.arguments as z.infer<typeof SipArraySipSchema>["arguments"];
      const offset = startCounter - 1;
      if (args.type === "array" && Array.isArray(args.value)) {
        return Float64Array.from(args.value.slice(offset, offset + count));
      }
      if (!this.externalLoader || !args.url || typeof args.value !== "string") {
        throw new Error(
          `ExternalArrayLoader is required to load external SIP_Array "${sip.name}"`
        );
      }
      const loaded = await this.externalLoader({
        type: args.type === "xlsx" ? "xlsx" : "csv",
        url: args.url,
        columnOrRange: args.value,
      });
      return loaded.subarray(offset, offset + count);
    }

    if (sip.function === "Lookup_Table") {
      const pairs = (sip.arguments as z.infer<typeof LookupTableSipSchema>["arguments"]).value;
      const cutoffs = Float64Array.from(pairs, ([cut]) => cut);
      const values = Float64Array.from(pairs, ([, val]) => val);

      if (sip.ref.source === "rng") {
        const [e, v, s3, s4] = this.resolveHdrSeeds(sip.ref.name, params);
        return this.wasm.hdrLookupBatch(
          startCounter,
          count,
          e,
          v,
          s3,
          s4,
          cutoffs,
          values
        );
      }
      const u = copulaLayers.get(`${sip.ref.name}:${sip.ref.copulaLayer}`);
      if (!u) {
        throw new Error(
          `Missing copula layer "${sip.ref.copulaLayer}" on "${sip.ref.name}"`
        );
      }
      return this.wasm.lookupBatch(u, cutoffs, values);
    }

    if (
      sip.function === "Metalog_1_0" ||
      sip.function === "Metalog_2_0" ||
      sip.function === "GeneralizedMetalog"
    ) {
      const aCoeffs =
        sip.function === "GeneralizedMetalog"
          ? this.resolveGeneralizedCoefficients(
              sip as z.infer<typeof GeneralizedMetalogSipSchema>,
              params
            )
          : Float64Array.from(
              (sip as z.infer<typeof MetalogSipSchema>).arguments.aCoefficients
            );

      const { lowerBound, upperBound } = (
        sip as z.infer<typeof MetalogSipSchema>
      ).arguments;
      const rounding = toWasmRounding(
        (sip as z.infer<typeof MetalogSipSchema>).rounding
      );

      if (sip.ref.source === "rng") {
        // Fused Rust/WASM path: HDR -> Metalog -> Rounding in a single pass
        const [e, v, s3, s4] = this.resolveHdrSeeds(sip.ref.name, params);
        return this.wasm.hdrMetalogBatch(
          startCounter,
          count,
          e,
          v,
          s3,
          s4,
          aCoeffs,
          lowerBound,
          upperBound,
          rounding
        );
      }

      const u = copulaLayers.get(`${sip.ref.name}:${sip.ref.copulaLayer}`);
      if (!u) {
        throw new Error(
          `Missing copula layer "${sip.ref.copulaLayer}" on "${sip.ref.name}"`
        );
      }
      return this.wasm.metalogBatch(u, aCoeffs, lowerBound, upperBound, rounding);
    }

    throw new Error(
      `Unsupported portable SIP function "${sip.function}" on "${sip.name}"`
    );
  }

  /**
   * Evaluates all portable SIPs in the library in bulk.
   * Shared copulas are computed exactly once across all dependent SIPs.
   */
  public async evaluateAll(
    options: EvaluateOptions
  ): Promise<Record<string, Float64Array>> {
    const startCounter = options.startCounter ?? 1;
    const params = options.parameters ?? {};

    const neededCopulas = new Set<string>();
    for (const sip of this.portableSips.values()) {
      if (sip.ref.source === "copula") {
        neededCopulas.add(sip.ref.name);
      }
    }

    const copulaLayers = await this.evaluateCopulaLayers(
      neededCopulas,
      startCounter,
      options.count,
      params
    );

    const results: Record<string, Float64Array> = {};
    for (const [name, sip] of this.portableSips.entries()) {
      results[name] = await this.evaluateSingleSip(
        sip,
        startCounter,
        options.count,
        params,
        copulaLayers
      );
    }
    return results;
  }

  /**
   * Evaluates a single SIP by name.
   */
  public async evaluateSip(
    name: string,
    options: EvaluateOptions
  ): Promise<Float64Array> {
    const sip = this.portableSips.get(name);
    if (!sip) {
      throw new Error(`SIP "${name}" not found in model "${this.model.name}"`);
    }
    const startCounter = options.startCounter ?? 1;
    const params = options.parameters ?? {};

    const neededCopulas = new Set<string>();
    if (sip.ref.source === "copula") {
      neededCopulas.add(sip.ref.name);
    }
    const copulaLayers = await this.evaluateCopulaLayers(
      neededCopulas,
      startCounter,
      options.count,
      params
    );
    return this.evaluateSingleSip(sip, startCounter, options.count, params, copulaLayers);
  }

  /**
   * Computes Welford summary statistics for a SIP.
   * For independent HDR-driven Metalogs, uses the O(1)-memory streaming WASM path
   * so millions of trials can be summarized without allocating a JS `Float64Array`.
   */
  public async summarizeSip(
    name: string,
    options: EvaluateOptions
  ): Promise<TrialStats> {
    const sip = this.portableSips.get(name);
    if (!sip) {
      throw new Error(`SIP "${name}" not found in model "${this.model.name}"`);
    }
    const startCounter = options.startCounter ?? 1;
    const params = options.parameters ?? {};

    if (
      sip.ref.source === "rng" &&
      (sip.function === "Metalog_1_0" ||
        sip.function === "Metalog_2_0" ||
        sip.function === "GeneralizedMetalog")
    ) {
      const aCoeffs =
        sip.function === "GeneralizedMetalog"
          ? this.resolveGeneralizedCoefficients(
              sip as z.infer<typeof GeneralizedMetalogSipSchema>,
              params
            )
          : Float64Array.from(
              (sip as z.infer<typeof MetalogSipSchema>).arguments.aCoefficients
            );

      const { lowerBound, upperBound } = (
        sip as z.infer<typeof MetalogSipSchema>
      ).arguments;
      const rounding = toWasmRounding(
        (sip as z.infer<typeof MetalogSipSchema>).rounding
      );
      const [e, v, s3, s4] = this.resolveHdrSeeds(sip.ref.name, params);

      return materializeStats(
        this.wasm.hdrMetalogStats(
          startCounter,
          options.count,
          e,
          v,
          s3,
          s4,
          aCoeffs,
          lowerBound,
          upperBound,
          rounding
        )
      );
    }

    const trials = await this.evaluateSip(name, options);
    return materializeStats(this.wasm.summarizeTrials(trials));
  }
}