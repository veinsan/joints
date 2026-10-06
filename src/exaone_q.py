"""EXAONE-Tabular regressor with the full predictive distribution (the package only returns a point estimate).

The released regressor emits 999 quantiles per ensemble member (tau = 0.001 ... 0.999); predict() collapses
them to a trimmed mean. predict_quantiles() keeps them: per member the requested taus are read off the sorted
quantile bank, de-normalised, and averaged over members (Vincentisation, uniform member weights).
"""
import numpy as np
import torch
from exaonetabular import EXAONETabularRegressor


class EXAONEQuantileRegressor(EXAONETabularRegressor):
    _taus = None

    def _collapse_members(self, output, query_count):
        if self._taus is None:
            return super()._collapse_members(output, query_count)
        q = torch.sort(output.float(), dim=-1).values
        n = q.shape[-1]
        idx = torch.as_tensor(np.clip(np.round(np.asarray(self._taus) * (n + 1)).astype(int) - 1, 0, n - 1), device=q.device)
        return q[..., idx]                                          # (members, rows, len(taus))

    def predict_quantiles(self, features, taus):
        st = self._state()
        query = st["preprocessor"].transform(features).values
        self._taus = list(taus)
        try:
            mp = self._pooled_member_points(
                torch.as_tensor(st["support_x"], dtype=torch.float32, device=self.device),
                torch.as_tensor(st["support_y"], dtype=torch.float32, device=self.device),
                torch.as_tensor(query, dtype=torch.float32, device=self.device), st["passes"])
        finally:
            self._taus = None
        q = (mp * st["scale"] + st["center"]).mean(dim=0)
        return np.sort(q.cpu().numpy().astype(np.float64), axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)); y = X[:, 0] * 2 + rng.normal(size=600)
    m = EXAONEQuantileRegressor.from_pretrained(device="cpu", ensemble_count=2, compute_dtype="float32", seed=0).fit(X[:500], y[:500])
    Q = m.predict_quantiles(X[500:], [0.1, 0.5, 0.9])
    p = m.predict(X[500:])
    cover = np.mean((y[500:] > Q[:, 0]) & (y[500:] < Q[:, 2]))
    print("quantile shape", Q.shape, "| 80% interval coverage", round(cover, 2), "| corr(median, point)", round(np.corrcoef(Q[:, 1], p)[0, 1], 3))
    assert Q.shape == (100, 3) and 0.6 < cover < 0.95 and np.corrcoef(Q[:, 1], p)[0, 1] > 0.95
