"""Generalised linear mixed models with one random intercept per slide, by maximum likelihood.

The models of the official analysis for counts and shares (WT3 against GD1), as a check of the linear mixed model
on log(1 + y) that the notebook reports:

* **negative binomial** (NB2: variance μ + μ²/θ) or **Poisson** for a count, log link, with log(exposure) as an
  offset (e.g. crossed cells with the junction length): the effect exp(β) is a rate ratio;
* **binomial** for a share (successes out of trials, e.g. crossed out of crossed + home), logit link, optionally
  with an observation-level random intercept (one per section) for overdispersion: exp(β) is an odds ratio.

The slide's random intercept b ~ N(0, σ²) is on the scale of the link. Each slide's likelihood is integrated over b
by adaptive Gauss–Hermite quadrature (nodes centred on the mode of the slide's integrand and scaled by its
curvature); the observation-level effect of the binomial model is integrated the same way inside it. The
parameters (β, log σ, and log θ or log τ) are found by BFGS on the log-likelihood with central-difference
gradients, and the Wald standard errors come from a finite-difference Hessian of the log-likelihood.

This is the reference implementation that the figure builder's own (JavaScript, figure_builder.html) is checked
against. Pure numpy / scipy, no Qt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

#: Quadrature nodes per slide.
NODES = 25
#: Quadrature nodes per section for the observation-level random effect (binomial with overdispersion).
INNER_NODES = 15
#: Below this the slide SD counts as zero (the boundary of the parameter space).
SIGMA_ZERO = 1e-3
#: Families this module fits.
FAMILIES = ("nb", "poisson", "binomial")


def log_density(family: str, eta, y, n=None, theta=None):
    """log f(y | η) and its first two derivatives in η, for one family (arrays broadcast)."""
    if family == "poisson":
        mu = np.exp(eta)
        return y * eta - mu - special.gammaln(y + 1), y - mu, -mu
    if family == "nb":
        log_sum = np.logaddexp(np.log(theta), eta)              # log(θ + μ)
        q = np.exp(eta - log_sum)                               # μ / (θ + μ)
        lf = (special.gammaln(y + theta) - special.gammaln(theta) - special.gammaln(y + 1)
              + theta * (np.log(theta) - log_sum) + y * (eta - log_sum))
        return lf, y * (1 - q) - theta * q, -(y + theta) * q * (1 - q)
    if family == "binomial":
        q = special.expit(eta)
        lf = (special.gammaln(n + 1) - special.gammaln(y + 1) - special.gammaln(n - y + 1)
              + y * eta - n * np.logaddexp(0, eta))
        return lf, y - n * q, -n * q * (1 - q)
    raise ValueError(f"unknown family {family!r}")


def _olre_density(eta, y, n, tau, inner):
    """Binomial with u ~ N(0, τ²) added to η, integrated over u: log f and its first two derivatives in η."""
    z, w = inner
    u = np.zeros_like(eta)
    for _ in range(100):
        _, d1, d2 = log_density("binomial", eta + u, y, n)
        step = np.clip(-(d1 - u / tau ** 2) / (d2 - 1 / tau ** 2), -3, 3)
        u = u + step
        if np.max(np.abs(step)) < 1e-11:
            break
    _, _, d2 = log_density("binomial", eta + u, y, n)
    scale = np.sqrt(2 / (1 / tau ** 2 - d2))
    uk = u[..., None] + scale[..., None] * z
    lf, d1, d2 = log_density("binomial", eta[..., None] + uk, y[..., None], n[..., None])
    h = lf - uk ** 2 / (2 * tau ** 2) + np.log(w) + z ** 2
    top = h.max(axis=-1, keepdims=True)
    wt = np.exp(h - top)
    total = wt.sum(axis=-1)
    post = wt / total[..., None]
    m1 = (post * d1).sum(-1)
    log_f = top[..., 0] + np.log(total) + np.log(scale) - 0.5 * np.log(2 * np.pi * tau ** 2)
    return log_f, m1, (post * d2).sum(-1) + (post * d1 ** 2).sum(-1) - m1 ** 2


class _Model:
    def __init__(self, y, X, groups, family, offset=None, trials=None, olre=False, nodes=NODES,
                 inner_nodes=INNER_NODES):
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.family, self.olre = family, bool(olre) and family == "binomial"
        self.offset = np.zeros(len(self.y)) if offset is None else np.asarray(offset, float)
        self.trials = np.ones(len(self.y)) if trials is None else np.asarray(trials, float)
        self.labels, self.g = np.unique(np.asarray(groups), return_inverse=True)
        self.G = len(self.labels)
        self.onehot = np.zeros((self.G, len(self.y)))
        self.onehot[self.g, np.arange(len(self.y))] = 1.0
        self.z, self.w = np.polynomial.hermite.hermgauss(nodes)
        self.inner = np.polynomial.hermite.hermgauss(inner_nodes)
        self.p = self.X.shape[1]
        self.extra = family == "nb" or self.olre

    def density(self, eta, y, n, extra):
        if self.olre:
            return _olre_density(eta, y, n, extra, self.inner)
        return log_density(self.family, eta, y, n, extra)

    def loglik(self, params) -> float:
        p = self.p
        beta, sigma2 = params[:p], np.exp(2 * params[p])
        extra = np.exp(params[p + 1]) if self.extra else None
        eta = self.X @ beta + self.offset
        y, n = self.y, self.trials
        b = np.zeros(self.G)
        for _ in range(200):                                     # each slide's mode (Newton, log-concave)
            _, d1, d2 = self.density(eta + b[self.g], y, n, extra)
            step = np.clip(-(self.onehot @ d1 - b / sigma2) / (self.onehot @ d2 - 1 / sigma2), -2, 2)
            b = b + step
            if np.max(np.abs(step)) < 1e-11:
                break
        _, _, d2 = self.density(eta + b[self.g], y, n, extra)
        scale = np.sqrt(2 / (1 / sigma2 - self.onehot @ d2))
        bk = b[:, None] + scale[:, None] * self.z                # (slides, nodes)
        lf, _, _ = self.density(eta[:, None] + bk[self.g], y[:, None], n[:, None], extra)
        h = self.onehot @ lf - bk ** 2 / (2 * sigma2) - 0.5 * np.log(2 * np.pi * sigma2) + np.log(self.w) + self.z ** 2
        return float(np.sum(np.log(scale) + special.logsumexp(h, axis=1)))

    def start(self):
        import statsmodels.api as sm
        if self.family == "binomial":
            frac = self.y / self.trials
            glm = sm.GLM(frac, self.X, family=sm.families.Binomial(), var_weights=self.trials).fit()
        else:
            glm = sm.GLM(self.y, self.X, family=sm.families.Poisson(), offset=self.offset).fit()
        return np.r_[glm.params, np.log(0.5), [0.0] if self.extra else []]


def _gradient(f, x, h=1e-5):
    g = np.zeros(len(x))
    for k in range(len(x)):
        e = np.zeros(len(x))
        e[k] = h
        g[k] = (f(x + e) - f(x - e)) / (2 * h)
    return g


def _hessian(f, x, h=1e-4):
    """Second derivatives of f from function values (central differences)."""
    m = len(x)
    H = np.zeros((m, m))
    f0 = f(x)
    for j in range(m):
        ej = np.zeros(m)
        ej[j] = h
        H[j, j] = (f(x + 2 * ej) - 2 * f0 + f(x - 2 * ej)) / (4 * h * h)
        for k in range(j):
            ek = np.zeros(m)
            ek[k] = h
            H[j, k] = H[k, j] = (f(x + ej + ek) - f(x + ej - ek) - f(x - ej + ek) + f(x - ej - ek)) / (4 * h * h)
    return H


def fit(y, X, groups, family, offset=None, trials=None, olre=False, nodes=NODES, inner_nodes=INNER_NODES,
        names=None) -> dict:
    """Fit the GLMM; returns the estimates, their Wald covariance, σ (slide), θ (NB) or τ (observation level),
    the log-likelihood and whether the optimiser converged."""
    model = _Model(y, X, groups, family, offset, trials, olre, nodes, inner_nodes)
    nll = lambda x: -model.loglik(x)
    res = optimize.minimize(nll, model.start(), jac=lambda x: _gradient(nll, x), method="BFGS",
                            options={"gtol": 1e-6, "maxiter": 1000})
    p = model.p
    sigma = float(np.exp(res.x[p]))
    # With the slide variance at its boundary (σ → 0) log σ has no curvature: the covariance is then that of the
    # other parameters.
    keep = [k for k in range(len(res.x)) if not (sigma < SIGMA_ZERO and k == p)]
    H = _hessian(nll, res.x)[np.ix_(keep, keep)]
    cov = np.linalg.inv(H)[:p, :p]
    gradient = float(np.max(np.abs(_gradient(nll, res.x))))
    names = list(names) if names is not None else [f"x{k}" for k in range(p)]
    return {"params": pd.Series(res.x[:p], index=names), "cov": pd.DataFrame(cov, index=names, columns=names),
            "se": pd.Series(np.sqrt(np.diag(cov)), index=names), "sigma": sigma, "sigma at zero": sigma < SIGMA_ZERO,
            "theta": float(np.exp(res.x[p + 1])) if family == "nb" else np.nan,
            "tau": float(np.exp(res.x[p + 1])) if model.olre else np.nan,
            "llf": -float(res.fun), "converged": gradient < 1e-3 and bool(np.all(np.linalg.eigvalsh(H) > 0)),
            "gradient": gradient, "sections": len(model.y), "slides": model.G}


def design(frame: pd.DataFrame, reference: str = "GD1", other: str = "WT3", experiment: bool = True):
    """Intercept, genotype (1 = *other*) and, if asked and more than one, experiment dummies (first = reference)."""
    cols = {"Intercept": np.ones(len(frame)), other: (frame["Genotype"] == other).to_numpy(float)}
    if experiment:
        levels = sorted(frame["Experiment"].astype(int).unique())
        for e in levels[1:]:
            cols[f"experiment {e}"] = (frame["Experiment"].astype(int) == e).to_numpy(float)
    X = pd.DataFrame(cols, index=frame.index)
    return X.to_numpy(), list(X.columns)


def genotype_effect(result: dict, term: str = "WT3") -> dict:
    """exp(β) of the genotype term with its Wald 95 % CI and two-sided P (normal)."""
    b, se = result["params"][term], result["se"][term]
    z = stats.norm.ppf(0.975)
    return {"effect": np.exp(b), "CI low": np.exp(b - z * se), "CI high": np.exp(b + z * se),
            "P": 2 * stats.norm.sf(abs(b / se)), "log effect": b, "SE": se}
