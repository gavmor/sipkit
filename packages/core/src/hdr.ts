/**
 * HDR 2.0 counter-based PRNG (ProbabilityManagement.org HDR_2_0 Seed Standard).
 *
 * Invariant: with every input an integer in [0, MAX_SEED], every intermediate
 * is an integer below 2^53, so plain Number arithmetic is exact and the result
 * matches the reference spreadsheet bit-for-bit.
 */

/** Largest admissible counter or seed (8 decimal digits). */
export const MAX_SEED = 99_999_999;

export interface HdrSeeds {
  entity: number;
  varId: number;
  seed3?: number;
  seed4?: number;
}

const K = 999_999_999_999_989;
const TWO_32 = 4_294_967_296;

function check(name: string, value: number): void {
  if (!Number.isInteger(value) || value < 0 || value > MAX_SEED) {
    throw new RangeError(`${name} must be an integer in [0, ${MAX_SEED}], got ${value}`);
  }
}

function checkSeeds(seeds: HdrSeeds): void {
  check('entity', seeds.entity);
  check('varId', seeds.varId);
  check('seed3', seeds.seed3 ?? 0);
  check('seed4', seeds.seed4 ?? 0);
}

/** Unchecked kernel; caller guarantees every input is in-domain. */
function hdr(counter: number, entity: number, varId: number, seed3: number, seed4: number): number {
  const x1 =
    (counter * 2499997 + varId * 1800451 + entity * 2000371 + seed3 * 1796777 + seed4 * 2299603) %
    7450589;
  const t1 = ((K % (x1 * 4658 + 7450581)) * 383) % 99991;
  const x2 =
    (counter * 2246527 + varId * 2399993 + entity * 2100869 + seed3 * 1918303 + seed4 * 1624729) %
    7450987;
  const t2 = ((K % (x2 * 7580 + 7560584)) * 17669) % 7440893;
  return ((((t1 * 7440893 + t2) * 1343) % TWO_32) + 0.5) / TWO_32;
}

/** Uniform in (0, 1) for trial `counter`. Throws RangeError for out-of-domain inputs. */
export function hdrUniform(counter: number, seeds: HdrSeeds): number {
  check('counter', counter);
  checkSeeds(seeds);
  return hdr(counter, seeds.entity, seeds.varId, seeds.seed3 ?? 0, seeds.seed4 ?? 0);
}

/** `out[i] = hdrUniform(firstCounter + i, seeds)` for i in [0, count). */
export function hdrFill(seeds: HdrSeeds, firstCounter: number, count: number): Float64Array {
  checkSeeds(seeds);
  check('firstCounter', firstCounter);
  if (!Number.isInteger(count) || count < 0) {
    throw new RangeError(`count must be a non-negative integer, got ${count}`);
  }
  if (count > 0) check('last counter', firstCounter + count - 1);
  const { entity, varId } = seeds;
  const seed3 = seeds.seed3 ?? 0;
  const seed4 = seeds.seed4 ?? 0;
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) out[i] = hdr(firstCounter + i, entity, varId, seed3, seed4);
  return out;
}
