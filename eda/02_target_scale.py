"""02 - Target & MASE-scale anatomy: what does the metric actually reward?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import KEY, fig_dir, load, release_dates, scale, simulate, style

plt = style()
F = fig_dir("02_target")
d = load()
tr, th = d["train"], d["hist"]

hs, ts = simulate(tr, release_dates(tr))
s_tr, s_te = scale(hs), scale(th)
print(f"simulated train: {ts.movie_title.nunique()} films, {len(s_tr)} pairs, {len(ts)} target rows")
print(f"test          : {th.movie_title.nunique()} films, {len(th.groupby(KEY))} history pairs")

print("\n=== scale s_p = max(mean D1-D3, 1) ===")
q = [.01, .05, .25, .5, .75, .95, .99]
te_pairs = d["test"][KEY].drop_duplicates()
s_te = s_te.reindex(pd.MultiIndex.from_frame(te_pairs))
print(pd.DataFrame({"train_sim": s_tr.quantile(q), "test": s_te.quantile(q)}).round(1))
for n, s in [("train_sim", s_tr), ("test", s_te)]:
    print(f"{n}: clipped at 1 = {(s == 1).mean():.3%}   s<5 = {(s < 5).mean():.2%}   s<20 = {(s < 20).mean():.2%}")

ts = ts.join(s_tr, on=KEY)
ts["r"] = ts.total_ticket / ts.scale
print("\n=== ratio r = y / s (what MASE actually scores) ===")
print(ts.r.describe(percentiles=q).round(3))
print("zero-target share:", (ts.total_ticket == 0).mean().round(3))
print("zero share by horizon:", ts.assign(z=ts.total_ticket == 0).groupby(ts.date_show.sub(ts.date_show.groupby([ts.movie_title, ts.cinema_ids]).transform('min')).dt.days + 4).z.mean().round(3).to_dict())

# weight of each scale bucket in the metric if we predicted 0 or predicted scale
b = pd.cut(ts.scale, [0, 1, 5, 20, 100, 500, 1e9], labels=["=1", "1-5", "5-20", "20-100", "100-500", ">500"], include_lowest=True)
g = ts.assign(b=b, err_naive=(ts.total_ticket - ts.scale).abs() / ts.scale).groupby("b", observed=True)
tab = pd.DataFrame({"rows": g.size(), "share": g.size() / len(ts), "median_r": g.r.median(),
                    "naive_mase": g.err_naive.mean(), "contrib": g.err_naive.sum() / len(ts)})
print("\n=== error mass by scale bucket (naive forecast yhat = s) ===")
print(tab.round(3))
print("overall naive (yhat = mean D1-D3) MASE:", round(ts.err_naive.mean() if 'err_naive' in ts else g.err_naive.sum().sum() / len(ts), 4))

# L1 optimum: global constant multiplier k minimising mean |r - k|  == median of r
k = np.linspace(0, 2, 201)
loss = [np.mean(np.abs(ts.r - kk)) for kk in k]
print(f"best constant multiplier k*={k[np.argmin(loss)]:.2f}  MASE={min(loss):.4f}   (median r = {ts.r.median():.3f})")
print(f"mean r = {ts.r.mean():.3f} -> MAE-optimal is the MEDIAN, mean is pulled by right tail")

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].hist(np.log10(s_tr), bins=60, alpha=.6, density=True, label="train_sim")
ax[0].hist(np.log10(s_te.dropna()), bins=60, alpha=.6, density=True, label="test")
ax[0].set(title="log10 scale s_p", xlabel="log10(s)"); ax[0].legend()
ax[1].hist(ts.r.clip(upper=4), bins=80, color="#2a78d6")
ax[1].axvline(ts.r.median(), color="#e34948", lw=1.5, label=f"median {ts.r.median():.2f}")
ax[1].set(title="r = y(D4..D10) / s  (clipped at 4)", xlabel="r"); ax[1].legend()
ax[2].plot(k, loss); ax[2].set(title="MASE of constant multiplier k*s", xlabel="k", ylabel="MASE")
fig.savefig(F / "scale_ratio.png")

# log-log scatter: pair scale vs mean D4-D10 ratio
pr = ts.groupby(KEY).agg(s=("scale", "first"), r=("r", "mean"))
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.scatter(np.log10(pr.s), pr.r.clip(upper=5), s=3, alpha=.25, color="#2a78d6")
m = pr.groupby(pd.cut(np.log10(pr.s), 20), observed=True).r.median()
ax.plot([i.mid for i in m.index], m.values, color="#e34948", label="median")
ax.set(title="Mean r per pair vs pair size", xlabel="log10 s", ylabel="mean r (clip 5)"); ax.legend()
fig.savefig(F / "ratio_vs_size.png")
print("\nmedian pair-mean r by log10(s) bin:\n", m.round(3).to_string())
print(f"figures -> {F}")
