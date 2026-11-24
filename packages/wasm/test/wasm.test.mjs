import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  hdrBatch,
  hdrLookupBatch,
  hdrMetalogBatch,
  hdrMetalogStats,
  initSync,
  lookupBatch,
  metalogBatch,
  summarizeTrials,
} from '../pkg/sipkit_wasm.js';

initSync({ module: readFileSync(new URL('../pkg/sipkit_wasm_bg.wasm', import.meta.url)) });

const fixture = (name) =>
  JSON.parse(readFileSync(new URL(`../../../fixtures/${name}`, import.meta.url), 'utf8'));

const MAX_SEED = 99_999_999;

test('hdrBatch reproduces the HDR 2.0 seed standard exactly', () => {
  const { seeds, cases } = fixture('hdr-2.0-seed-standard.json');
  assert.equal(cases.length, 10_000);
  const first = cases[0].counter;
  const out = hdrBatch(first, cases.length, seeds.entity, seeds.varId, seeds.seed3, seeds.seed4);
  assert.ok(out instanceof Float64Array);
  cases.forEach(({ counter, u }, i) => {
    assert.equal(counter, first + i);
    assert.equal(out[i], u, `counter ${counter}`);
  });
});

test('hdrBatch rejects non-integer and out-of-range inputs', () => {
  assert.throws(() => hdrBatch(MAX_SEED + 1, 1, 1, 2, 3, 4), /startCounter/);
  assert.throws(() => hdrBatch(1, 1, MAX_SEED + 1, 2, 3, 4), /entity/);
  assert.throws(() => hdrBatch(1.5, 1, 1, 2, 3, 4), /startCounter/);
  assert.throws(() => hdrBatch(1, 1, 1, -1, 3, 4), /varId/);
  assert.throws(() => hdrBatch(1, 1, 1, 2, Number.NaN, 4), /seed3/);
  assert.throws(() => hdrBatch(MAX_SEED, 2, 1, 2, 3, 4), /counter/);
  assert.throws(() => hdrBatch(1, 2.5, 1, 2, 3, 4), /count/);
  assert.ok(hdrBatch(MAX_SEED, 1, MAX_SEED, MAX_SEED, MAX_SEED, MAX_SEED)[0] > 0);
});

test('metalogBatch matches the quantile parity vectors', () => {
  for (const c of fixture('metalog-quantile.json').cases) {
    const ys = Float64Array.from(c.points, (p) => p.y);
    const batch = metalogBatch(ys, Float64Array.from(c.a), c.bounds.lower, c.bounds.upper, 0);
    c.points.forEach(({ y, x }, i) => {
      const tol = 1e-12 * Math.max(1, Math.abs(x));
      assert.ok(Math.abs(batch[i] - x) <= tol, `k=${c.a.length} ${c.bounds.kind} y=${y}`);
    });
    for (const y of [0, 1, -1, 2, Number.NaN]) {
      assert.ok(
        Number.isNaN(metalogBatch(Float64Array.of(y), c.a, c.bounds.lower, c.bounds.upper, 0)[0]),
      );
    }
  }
});

test('metalogBatch rejects invalid input', () => {
  const ys = Float64Array.of(0.5);
  assert.throws(() => metalogBatch(ys, Float64Array.of(1), undefined, undefined, 0));
  assert.throws(() => metalogBatch(ys, new Float64Array(17), undefined, undefined, 0));
  assert.throws(() => metalogBatch(ys, Float64Array.of(0, Number.NaN), undefined, undefined, 0));
  assert.throws(() => metalogBatch(ys, Float64Array.of(0, 1), 1, 1, 0));
  assert.throws(() =>
    metalogBatch(ys, Float64Array.of(0, 1), Number.POSITIVE_INFINITY, undefined, 0),
  );
});

test('hdrMetalogBatch equals metalogBatch(hdrBatch(...)) elementwise', () => {
  const [entity, varId, seed3, seed4, first, count] = [7, 11, 0, 0, 1, 5_000];
  for (const c of fixture('metalog-quantile.json').cases) {
    const sims = hdrMetalogBatch(
      first,
      count,
      entity,
      varId,
      seed3,
      seed4,
      Float64Array.from(c.a),
      c.bounds.lower,
      c.bounds.upper,
      0,
    );
    assert.equal(sims.length, count);
    const us = hdrBatch(first, count, entity, varId, seed3, seed4);
    const expected = metalogBatch(us, Float64Array.from(c.a), c.bounds.lower, c.bounds.upper, 0);
    assert.deepEqual([...sims], [...expected]);
  }
  assert.throws(() =>
    hdrMetalogBatch(MAX_SEED, 2, 1, 2, 3, 4, Float64Array.of(0, 1), undefined, undefined, 0),
  );
});

test('rounding modes match Excel ROUND / ROUNDUP / ROUNDDOWN', () => {
  // quantile(0.3) = ln(3/7) ≈ -0.847, quantile(0.7) ≈ +0.847 for a = [0, 1].
  const us = Float64Array.of(0.3, 0.7);
  const a = Float64Array.of(0, 1);
  const none = metalogBatch(us, a, undefined, undefined, 0);
  assert.ok(Math.abs(none[0] - Math.log(3 / 7)) < 1e-12);
  assert.ok(Math.abs(none[1] - Math.log(7 / 3)) < 1e-12);
  assert.deepEqual([...metalogBatch(us, a, undefined, undefined, 1)], [-1, 1]);
  assert.deepEqual([...metalogBatch(us, a, undefined, undefined, 2)], [-1, 1]);
  const down = metalogBatch(us, a, undefined, undefined, 3);
  assert.ok(down[0] === 0);
  assert.ok(down[1] === 0);
});

test('lookupBatch evaluates the discrete inverse CDF', () => {
  const cutoffs = Float64Array.of(0.25, 0.5, 1.0);
  const values = Float64Array.of(10, 20, 30);
  assert.deepEqual(
    [...lookupBatch(Float64Array.of(0, 0.25, 0.2500001, 0.5, 0.9, 1.0), cutoffs, values)],
    [10, 10, 20, 20, 30, 30],
  );
  assert.ok(Number.isNaN(lookupBatch(Float64Array.of(Number.NaN), cutoffs, values)[0]));
  assert.throws(() => lookupBatch(Float64Array.of(0.5), new Float64Array(0), values));
  assert.throws(() => lookupBatch(Float64Array.of(0.5), cutoffs, Float64Array.of(10, 20)));
  assert.throws(() =>
    lookupBatch(Float64Array.of(0.5), Float64Array.of(0.5, 0.25, 1.0), values),
  );
  assert.throws(() => lookupBatch(Float64Array.of(0.5), Float64Array.of(1.5), values));
});

test('hdrLookupBatch equals lookupBatch(hdrBatch(...))', () => {
  const cutoffs = Float64Array.of(0.25, 0.5, 1.0);
  const values = Float64Array.of(10, 20, 30);
  const us = hdrBatch(1, 5_000, 7, 11, 0, 0);
  assert.deepEqual(
    [...hdrLookupBatch(1, 5_000, 7, 11, 0, 0, cutoffs, values)],
    [...lookupBatch(us, cutoffs, values)],
  );
});

test('hdrMetalogStats streams Welford summaries without allocating trials', () => {
  const stats = hdrMetalogStats(1, 10_000, 1, 2, 3, 4, Float64Array.of(0, 1), undefined, undefined, 0);
  const batch = hdrMetalogBatch(1, 10_000, 1, 2, 3, 4, Float64Array.of(0, 1), undefined, undefined, 0);
  const mean = batch.reduce((s, v) => s + v, 0) / batch.length;
  assert.equal(stats.count, 10_000);
  assert.equal(stats.nanCount, 0);
  assert.ok(Math.abs(stats.mean - mean) < 1e-12);
  assert.equal(stats.min, Math.min(...batch));
  assert.equal(stats.max, Math.max(...batch));
  stats.free();
});

test('summarizeTrials computes sample moments and merges chunks', () => {
  const s = summarizeTrials(Float64Array.of(1, 2, 3, 4, 5));
  assert.equal(s.count, 5);
  assert.equal(s.nanCount, 0);
  assert.equal(s.mean, 3);
  assert.equal(s.variance, 2.5);
  assert.equal(s.stdDev, Math.sqrt(2.5));
  assert.equal(s.min, 1);
  assert.equal(s.max, 5);

  const withNan = summarizeTrials(Float64Array.of(1, Number.NaN, 3));
  assert.equal(withNan.count, 2);
  assert.equal(withNan.nanCount, 1);
  assert.equal(withNan.mean, 2);

  const merged = s.merge(summarizeTrials(Float64Array.of(6, 7, 8, 9, 10)));
  assert.equal(merged.count, 10);
  assert.equal(merged.mean, 5.5);
  assert.ok(Math.abs(merged.variance - 82.5 / 9) < 1e-12);
  s.free();
  withNan.free();
  merged.free();
});
