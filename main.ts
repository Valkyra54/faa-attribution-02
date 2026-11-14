// w720b06-hl20: per-market quantile forecast for Polymarket Yes prices.
//
// For each market the forecast blends, at every horizon, the empirical
// quantiles of the last 720 one-minute moves with a normal shape whose scale
// comes from a 20-minute half-life EWMA-RMS. Both terms widen with the same
// h^0.4 growth. The median is not pinned: it follows the blended median
// move. No drift, no cross-market pull, no shared normaliser.
'use strict';

const fs = require('fs');
const path = require('path');

const TAU = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const WINDOW_MOVES = 720;
const BLEND = 0.6;
const HALF_LIFE_MIN = 20;
const ANCHOR = 0.0025;
const SIGMA_MIN = 0.001;
const SIGMA_MAX = 0.012;
const SHORT_HISTORY_MOVES = 60;
const DEFAULT_HORIZON = 240;

// Type-7 interpolated quantile of an ascending array at probability p.
function type7Quantile(sorted, p) {
  const n = sorted.length;
  if (n === 0) return 0;
  if (n === 1) return sorted[0];
  const pos = (n - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (pos - lo) * (sorted[hi] - sorted[lo]);
}

// The last WINDOW_MOVES one-minute differences of a price series.
function recentMoves(prices) {
  const diffs = [];
  for (let i = 1; i < prices.length; i++) diffs.push(prices[i] - prices[i - 1]);
  return diffs.length > WINDOW_MOVES ? diffs.slice(diffs.length - WINDOW_MOVES) : diffs;
}

// Root mean square of the moves, weighted so the newest move counts 1 and
// each older move fades by 0.5^(1/20) per minute of age (20-minute half-life).
function ewmaRms(moves) {
  if (moves.length === 0) return 0;
  const decay = Math.pow(0.5, 1 / HALF_LIFE_MIN);
  let weightSum = 0;
  let weightedSq = 0;
  let w = 1;
  for (let i = moves.length - 1; i >= 0; i--) {
    weightedSq += w * moves[i] * moves[i];
    weightSum += w;
    w *= decay;
  }
  return Math.sqrt(weightedSq / weightSum);
}

// One market's full 240-row forecast grid: five quantiles per horizon.
function forecastMarket(prices, horizon) {
  const anchor = prices[prices.length - 1];
  const moves = recentMoves(prices);
  const ordered = moves.slice().sort((a, b) => a - b);
  const empirical = TAU.map((t) => type7Quantile(ordered, t));
  const rawSigma = 0.5 * ewmaRms(moves) + 0.5 * ANCHOR;
  const sigma = Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, rawSigma));
  // On short histories the empirical weight fades toward zero and the
  // normal term carries the forecast, so a young market never leans on a
  // handful of moves.
  const blend = BLEND * Math.min(1, moves.length / SHORT_HISTORY_MOVES);
  const rows = [];
  for (let h = 1; h <= horizon; h++) {
    const growth = Math.pow(h, 0.4);
    const row = [];
    let running = 0;
    for (let j = 0; j < TAU.length; j++) {
      let q = anchor + blend * empirical[j] * growth + (1 - blend) * Z[j] * sigma * growth;
      q = Math.round(q * 1e6) / 1e6 + 0;
      if (q < 0) q = 0;
      if (q > 1) q = 1;
      if (q < running) q = running;
      running = q;
      row.push(q);
    }
    rows.push(row);
  }
  return rows;
}

function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR || '/input';
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || '/output';
  fs.mkdirSync(outDir, { recursive: true });
  const files = fs.readdirSync(inDir).filter((f) => f.endsWith('.in')).sort();
  for (const file of files) {
    const kase = JSON.parse(fs.readFileSync(path.join(inDir, file), 'utf8'));
    const horizon = Number.isInteger(kase.horizon_minutes) && kase.horizon_minutes > 0
      ? kase.horizon_minutes
      : DEFAULT_HORIZON;
    const forecasts = (kase.markets || []).map((market) => ({
      id: market.id,
      q: forecastMarket(market.prices || [], horizon),
    }));
    const outName = file.replace(/\.in$/, '.out');
    fs.writeFileSync(path.join(outDir, outName), JSON.stringify({ forecasts }));
  }
}

main();
