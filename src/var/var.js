// VaR and expected shortfall in the browser. Mirrors src/var/var.py; tests compare the two.
// Losses are positive fractions of the portfolio's value.

// Inverse normal CDF (Acklam's rational approximation, refined with one Halley step).
function normInv(p) {
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425, hi = 1 - lo;
  let x;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= hi) {
    const q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const e = 0.5 * erfc(-x / Math.SQRT2) - p;  // Halley refinement
  const u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
  return x - u / (1 + x * u / 2);
}

function erfc(x) {  // Numerical Recipes erfc, relative error < 1.2e-7
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

const normPdf = (z) => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);

function quantile(sorted, p) {  // numpy's default (linear interpolation)
  const i = p * (sorted.length - 1), lo = Math.floor(i), f = i - lo;
  return lo + 1 < sorted.length ? sorted[lo] + f * (sorted[lo + 1] - sorted[lo]) : sorted[lo];
}

function historical(r, alpha) {
  const s = [...r].sort((x, y) => x - y), q = quantile(s, 1 - alpha);
  const tail = s.filter((x) => x <= q);
  return [-q, -tail.reduce((a, b) => a + b, 0) / tail.length];
}

function moments(r) {  // mean, sample sd (ddof 1), skew and excess kurtosis as scipy (biased)
  const n = r.length, mean = r.reduce((a, b) => a + b, 0) / n;
  let m2 = 0, m3 = 0, m4 = 0;
  for (const x of r) { const d = x - mean; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; }
  m2 /= n; m3 /= n; m4 /= n;
  return { mean, sd: Math.sqrt(m2 * n / (n - 1)), skew: m3 / m2 ** 1.5, kurt: m4 / (m2 * m2) - 3 };
}

function normal(mu, sd, alpha) {
  const z = normInv(1 - alpha);
  return [-(mu + sd * z), -mu + sd * normPdf(z) / (1 - alpha)];
}

const cfZ = (z, s, k) => z + (z * z - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s * s / 36;

function cornishFisher(mu, sd, s, k, alpha) {
  const v = -(mu + sd * cfZ(normInv(1 - alpha), s, k));
  let tot = 0;
  for (let i = 0; i < 2000; i++) tot += cfZ(normInv((i + 0.5) / 2000 * (1 - alpha)), s, k);
  return [v, -(mu + sd * tot / 2000)];
}

function cfValid(s, k) {  // the expansion must keep rising in z to be a quantile function
  let prev = cfZ(-4, s, k);
  for (let i = 1; i <= 800; i++) { const v = cfZ(-4 + i * 0.01, s, k); if (!(v > prev)) return false; prev = v; }
  return true;
}

function horizon(r, days) {  // compound returns over overlapping windows of `days`
  const g = []; let acc = 1;
  for (const x of r) { acc *= 1 + x; g.push(acc); }
  const out = [g[days - 1] - 1];
  for (let i = days; i < g.length; i++) out.push(g[i] / g[i - days] - 1);
  return out;
}

function portfolio(returns, holdings) {  // today's holdings under each past day; skip gaps
  const names = Object.keys(holdings).filter((k) => holdings[k] > 0);
  const total = names.reduce((a, k) => a + holdings[k], 0);
  const n = returns[names[0]].length, out = [];
  for (let i = 0; i < n; i++) {
    let r = 0, ok = true;
    for (const k of names) { const x = returns[k][i]; if (x === null || x === undefined) { ok = false; break; } r += x * holdings[k] / total; }
    if (ok) out.push(r);
  }
  return out;
}

if (typeof module !== "undefined") module.exports = { normInv, historical, moments, normal, cornishFisher, cfValid, horizon, portfolio };
