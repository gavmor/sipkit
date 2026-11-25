import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  choleskyDecompose,
  gaussianCopulaBatch,
  initSync,
  metalogBatch,
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

const fixture = (name) =>
  JSON.parse(readFileSync(new URL(`../../../fixtures/${name}`, import.meta.url), 'utf8'));

// Base seeds from the parity fixture; dimension i draws with varId + i,
// matching GaussianCopula::sample_uniforms in the Rust kernel.
const packSeeds = (dim, { entity, varId, seed3, seed4 }) => {
  const flat = new Uint32Array(dim * 4);
  for (let i = 0; i < dim; i++) {
    flat[i * 4] = entity;
    flat[i * 4 + 1] = varId + i;
    flat[i * 4 + 2] = seed3;
    flat[i * 4 + 3] = seed4;
  }
  return flat;
};

test('normalQuantile matches known points', () => {
  assert.equal(normalQuantile(0.5), 0);
  assert.ok(Math.abs(normalQuantile(0.975) - 1.959963984540054) < 5e-9);
  assert.ok(Math.abs(normalQuantile(0.001) + 3.090232306167813) < 1e-7);
  for (const p of [0, 1, -0.1, 1.1, Number.NaN]) assert.ok(Number.isNaN(normalQuantile(p)));
  for (const z of [-2.5, -1, 0, 1, 2.5]) {
    assert.ok(Math.abs(normalQuantile(normalCdf(z)) - z) < 1e-5, `z=${z}`);
  }
});

test('choleskyDecompose factors and reconstructs, rejects non-PSD', () => {
  const L = choleskyDecompose(2, [4, 2, 2, 3]);
  assert.deepEqual([...L], [2, 0, 1, Math.SQRT2]);
  const R = [1, 0.8, 0.3, 0.8, 1, 0.5, 0.3, 0.5, 1];
  const L3 = choleskyDecompose(3, R);
  // L Lᵀ == R
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += L3[i * 3 + k] * L3[j * 3 + k];
      assert.ok(Math.abs(s - R[i * 3 + j]) < 1e-12);
    }
  }
  assert.throws(() => choleskyDecompose(2, [1, 2, 2, 1]));
  assert.throws(() => choleskyDecompose(2, [1, 0.5]));
  assert.throws(() => choleskyDecompose(2, [1, 0.5, 0.6, 1]));
  assert.throws(() => choleskyDecompose(0, []));
  assert.throws(() => choleskyDecompose(1.5, [1]));
});

test('gaussianCopulaBatch is reproducible and recovers ρ = 0.8', () => {
  const L = choleskyDecompose(2, [1, 0.8, 0.8, 1]);
  const seeds = packSeeds(2, { entity: 1, varId: 2, seed3: 3, seed4: 4 });
  const one = gaussianCopulaBatch(2, 7, 1, seeds, L);
  assert.deepEqual([...one], [...gaussianCopulaBatch(2, 7, 1, seeds, L)]);
  assert.notDeepEqual([...one], [...gaussianCopulaBatch(2, 8, 1, seeds, L)]);

  const n = 20_000;
  const out = gaussianCopulaBatch(2, 1, n, seeds, L);
  const us = [...out.subarray(0, n)].map((u) => normalQuantile(u));
  const vs = [...out.subarray(n, 2 * n)].map((u) => normalQuantile(u));
  const r = pearson(us, vs);
  assert.ok(Math.abs(r - 0.8) < 0.03, `r=${r}`);
});

test('gaussianCopulaBatch is column-major and matches single trials', () => {
  const L = choleskyDecompose(2, [1, 0.5, 0.5, 1]);
  const seeds = packSeeds(2, { entity: 1, varId: 2, seed3: 3, seed4: 4 });
  const batch = gaussianCopulaBatch(2, 42, 3, seeds, L);
  assert.equal(batch.length, 6);
  for (let t = 0; t < 3; t++) {
    const one = gaussianCopulaBatch(2, 42 + t, 1, seeds, L);
    assert.equal(batch[t], one[0]);
    assert.equal(batch[3 + t], one[1]);
  }
});

test('gaussianCopulaBatch preserves marginals and rejects bad input', () => {
  const L = choleskyDecompose(2, [1, 0.7, 0.7, 1]);
  const seeds = packSeeds(2, { entity: 1, varId: 2, seed3: 3, seed4: 4 });
  const flat = gaussianCopulaBatch(2, 1, 20_000, seeds, L);
  const xs = [...metalogBatch(flat.subarray(0, 20_000), [0, 1], undefined, undefined, 0)].sort(
    (a, b) => a - b,
  );
  const ys = [
    ...metalogBatch(flat.subarray(20_000, 40_000), [Math.log(100), 0.2], 0, undefined, 0),
  ].sort((a, b) => a - b);
  assert.ok(Math.abs(xs[10_000]) < 0.05);
  assert.ok(Math.abs(ys[10_000] - 100) < 1.5);
  assert.throws(() => gaussianCopulaBatch(2, 1, 1, new Uint32Array([1, 2, 3]), L));
  assert.throws(() => gaussianCopulaBatch(2, 1, 1, seeds, [1, 0]));
  assert.throws(() => gaussianCopulaBatch(0, 1, 1, new Uint32Array(0), []));
});

test('gaussianCopulaBatch matches the shared parity fixture', () => {
  // The Cholesky factor is exact (only +, -, *, / and sqrt). Correlated
  // uniforms and samples pass through ln/exp, which engines may round one
  // ulp apart, so they use the fixture's declared tolerance. The same file
  // is checked by the Rust suite.
  const parity = fixture('copula-parity.json');
  const close = (actual, expected, what) => {
    assert.equal(actual.length, expected.length, `${what} length`);
    actual.forEach((a, i) => {
      const e = expected[i];
      assert.ok(
        Math.abs(a - e) <= parity.value_tol * Math.max(1, Math.abs(e)),
        `${what}[${i}]: ${a} vs ${e}`,
      );
    });
  };
  for (const c of parity.cases) {
    const dim = c.marginals.length;
    const L = choleskyDecompose(dim, c.correlation.flat());
    assert.deepEqual([...L], c.factor.flat());
    const seeds = packSeeds(dim, c.seeds);
    for (const s of c.samples) {
      const uniforms = gaussianCopulaBatch(dim, s.counter, 1, seeds, L);
      close([...uniforms], s.uniforms, `counter ${s.counter} uniforms`);
      for (let d = 0; d < dim; d++) {
        const m = c.marginals[d];
        const x = metalogBatch(
          Float64Array.of(uniforms[d]),
          Float64Array.from(m.a),
          m.bounds.lower,
          m.bounds.upper,
          0,
        );
        close([x[0]], [s.x[d]], `counter ${s.counter} dim ${d}`);
      }
    }
  }
});

test('normalQuantile / normalCdf match the normal accuracy fixture', () => {
  const { quantile_tol, cdf_tol, quantile, cdf } = fixture('normal-accuracy.json');
  for (const { p, z } of quantile) {
    const err = Math.abs(normalQuantile(p) - z);
    assert.ok(err <= quantile_tol * Math.max(1, Math.abs(z)), `p=${p} err=${err}`);
  }
  for (const { x, p } of cdf) {
    assert.ok(Math.abs(normalCdf(x) - p) <= cdf_tol * p, `x=${x}`);
  }
});
