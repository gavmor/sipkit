import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { Metalog, hdrFill, hdrUniform, initSync } from '../pkg/sipkit_wasm.js';

initSync({ module: readFileSync(new URL('../pkg/sipkit_wasm_bg.wasm', import.meta.url)) });

const fixture = (name) =>
  JSON.parse(readFileSync(new URL(`../../../fixtures/${name}`, import.meta.url), 'utf8'));

const MAX_SEED = 99_999_999;

const metalogOf = ({ a, bounds }) =>
  new Metalog(Float64Array.from(a), bounds.lower, bounds.upper);

test('hdrFill reproduces the HDR 2.0 seed standard exactly', () => {
  const { seeds, cases } = fixture('hdr-2.0-seed-standard.json');
  assert.equal(cases.length, 10_000);
  const first = cases[0].counter;
  const out = hdrFill(seeds.entity, seeds.varId, seeds.seed3, seeds.seed4, first, cases.length);
  assert.ok(out instanceof Float64Array);
  cases.forEach(({ counter, u }, i) => {
    assert.equal(counter, first + i);
    assert.equal(out[i], u, `counter ${counter}`);
  });
});

test('hdrUniform and hdrFill reject non-integer and out-of-range inputs', () => {
  assert.throws(() => hdrUniform(MAX_SEED + 1, 1, 2, 3, 4), /counter/);
  assert.throws(() => hdrUniform(1, MAX_SEED + 1, 2, 3, 4), /entity/);
  assert.throws(() => hdrUniform(1.5, 1, 2, 3, 4), /counter/);
  assert.throws(() => hdrUniform(1, 1, -1, 3, 4), /varId/);
  assert.throws(() => hdrUniform(1, 1, 2, Number.NaN, 4), /seed3/);
  assert.throws(() => hdrFill(1, 2, 3, 4, MAX_SEED, 2), /counter/);
  assert.throws(() => hdrFill(1, 2, 3, 4, 1, 2.5), /count/);
  assert.ok(hdrUniform(MAX_SEED, MAX_SEED, MAX_SEED, MAX_SEED, MAX_SEED) > 0);
});

test('Metalog matches the quantile parity vectors', () => {
  for (const c of fixture('metalog-quantile.json').cases) {
    const m = metalogOf(c);
    const ys = Float64Array.from(c.points, (p) => p.y);
    const batch = m.quantiles(ys);
    c.points.forEach(({ y, x }, i) => {
      const tol = 1e-12 * Math.max(1, Math.abs(x));
      assert.ok(Math.abs(m.quantile(y) - x) <= tol, `k=${c.a.length} ${c.bounds.kind} y=${y}`);
      assert.equal(batch[i], m.quantile(y));
    });
    for (const y of [0, 1, -1, 2, Number.NaN]) assert.ok(Number.isNaN(m.quantile(y)));
    m.free();
  }
});

test('Metalog constructor rejects invalid input', () => {
  assert.throws(() => new Metalog(Float64Array.of(1)));
  assert.throws(() => new Metalog(new Float64Array(17)));
  assert.throws(() => new Metalog(Float64Array.of(0, Number.NaN)));
  assert.throws(() => new Metalog(Float64Array.of(0, 1), 1, 1));
  assert.throws(() => new Metalog(Float64Array.of(0, 1), Number.POSITIVE_INFINITY));
});

test('simulate equals quantile(hdrUniform(...)) elementwise', () => {
  const [entity, varId, seed3, seed4, first, count] = [7, 11, 0, 0, 1, 5_000];
  for (const c of fixture('metalog-quantile.json').cases) {
    const m = metalogOf(c);
    const sims = m.simulate(entity, varId, seed3, seed4, first, count);
    assert.equal(sims.length, count);
    for (let i = 0; i < count; i++) {
      assert.equal(sims[i], m.quantile(hdrUniform(first + i, entity, varId, seed3, seed4)));
    }
    m.free();
  }
  assert.throws(() => metalogOf({ a: [0, 1], bounds: {} }).simulate(1, 2, 3, 4, MAX_SEED, 2));
});
