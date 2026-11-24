import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  GaussianCopula,
  Metalog,
  cholesky,
  hdrUniform,
  initSync,
  normalCdf,
  normalQuantile,
} from '../pkg/sipkit_wasm.js';

initSync({ module: readFileSync(new URL('../pkg/sipkit_wasm_bg.wasm', import.meta.url)) });

const pearson = (xs, ys) => {
  const n = xs.length;
  const mx = xs.reduce((s, v) => s + v, 0) / n;
  const my = ys.reduce((s, v) => s + v, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  return sxy / Math.sqrt(sxx * syy);
};

const seeds = [1, 2, 3, 4]; // entity, varId, seed3, seed4

test('normalQuantile matches known points', () => {
  assert.equal(normalQuantile(0.5), 0);
  assert.ok(Math.abs(normalQuantile(0.975) - 1.959963984540054) < 5e-9);
  assert.ok(Math.abs(normalQuantile(0.001) + 3.090232306167813) < 1e-7);
  for (const p of [0, 1, -0.1, 1.1, Number.NaN]) assert.ok(Number.isNaN(normalQuantile(p)));
  for (const z of [-2.5, -1, 0, 1, 2.5]) {
    assert.ok(Math.abs(normalQuantile(normalCdf(z)) - z) < 1e-5, `z=${z}`);
  }
});

test('cholesky factors and reconstructs, rejects non-PSD', () => {
  const L = cholesky([4, 2, 2, 3], 2);
  assert.deepEqual([...L], [2, 0, 1, Math.SQRT2]);
  const R = [1, 0.8, 0.3, 0.8, 1, 0.5, 0.3, 0.5, 1];
  const L3 = cholesky(R, 3);
  // L Lᵀ == R
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += L3[i * 3 + k] * L3[j * 3 + k];
      assert.ok(Math.abs(s - R[i * 3 + j]) < 1e-12);
    }
  }
  assert.throws(() => cholesky([1, 2, 2, 1], 2));
  assert.throws(() => cholesky([1, 0.5], 2));
});

test('GaussianCopula is reproducible and recovers ρ = 0.8', () => {
  const cop = new GaussianCopula(
    [1, 0.8, 0.8, 1],
    2,
    [new Metalog([0, 1]), new Metalog([0, 1])],
  );
  assert.deepEqual(
    [...cop.sample(7, ...seeds)],
    [...cop.sample(7, ...seeds)],
  );
  assert.notDeepEqual(
    [...cop.sample(7, ...seeds)],
    [...cop.sample(8, ...seeds)],
  );
  const n = 20_000;
  const us = [];
  for (let c = 1; c <= n; c++) us.push([...cop.sampleUniforms(c, ...seeds)]);
  const r = pearson(
    us.map(([u]) => normalQuantile(u)),
    us.map(([, v]) => normalQuantile(v)),
  );
  assert.ok(Math.abs(r - 0.8) < 0.03, `r=${r}`);
  cop.free();
});

test('GaussianCopula sampleMany is flat row-major and matches sample()', () => {
  const cop = new GaussianCopula(
    [1, 0.5, 0.5, 1],
    2,
    [new Metalog([0, 1]), new Metalog([10, 2])],
  );
  const flat = cop.sampleMany(1, 100, ...seeds);
  assert.equal(flat.length, 200);
  for (let i = 0; i < 100; i++) {
    const one = cop.sample(1 + i, ...seeds);
    assert.equal(flat[2 * i], one[0]);
    assert.equal(flat[2 * i + 1], one[1]);
  }
  // sample() == marginal quantiles of sampleUniforms()
  const m1 = new Metalog([0, 1]);
  const m2 = new Metalog([10, 2]);
  const u = cop.sampleUniforms(42, ...seeds);
  const s = cop.sample(42, ...seeds);
  assert.equal(s[0], m1.quantile(u[0]));
  assert.equal(s[1], m2.quantile(u[1]));
  cop.free();
  m1.free();
  m2.free();
});

test('GaussianCopula preserves marginals and rejects bad input', () => {
  const cop = new GaussianCopula(
    [1, 0.7, 0.7, 1],
    2,
    [new Metalog([0, 1]), new Metalog([Math.log(100), 0.2], 0)],
  );
  const flat = cop.sampleMany(1, 20_000, ...seeds);
  const xs = [];
  const ys = [];
  for (let i = 0; i < 20_000; i++) {
    xs.push(flat[2 * i]);
    ys.push(flat[2 * i + 1]);
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  assert.ok(Math.abs(xs[10_000]) < 0.05);
  assert.ok(Math.abs(ys[10_000] - 100) < 1.5);
  cop.free();
  assert.throws(() => new GaussianCopula([2, 0, 0, 1], 2, [new Metalog([0, 1]), new Metalog([0, 1])]));
  assert.throws(() => new GaussianCopula([1, 0, 0, 1], 2, [new Metalog([0, 1])]));
});
