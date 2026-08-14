import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats
from scipy.optimize import minimize

plt.rcParams.update(
    {
        "figure.facecolor": "black",
        "axes.facecolor": "black",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "axes.edgecolor": "white",
        "font.family": "monospace",
        "font.monospace": ["Cascadia Code", "DejaVu Sans Mono", "Consolas"],
    }
)


def fit_best_count_model(data):
    """Fit candidate count distributions by maximum likelihood and return the
    one with the lowest AIC. AIC = 2k - 2*loglik rewards fit but penalizes each
    extra parameter, so it picks the appropriate shape *from the data* instead
    of you hardcoding it:
        - overdispersed / skewed (e.g. total bases) -> Negative Binomial
        - equidispersed / humped (e.g. strikeouts)   -> Poisson
    Returns (best_name, {name: {"aic": float, "pmf": callable}}).
    """
    data = np.rint(np.asarray(data)).astype(int)  # discrete counts
    m = data.mean()
    v = max(data.var(), 1e-9)
    cand = {}

    # --- Poisson (1 free param): MLE lambda = sample mean ---
    ll = stats.poisson.logpmf(data, m).sum()
    cand["Poisson"] = {
        "aic": 2 * 1 - 2 * ll,
        "pmf": lambda k, lam=m: stats.poisson.pmf(k, lam),
    }

    # --- Negative Binomial (2 free params): numerical MLE ---
    # Parameterized by (mu, log r) for a stable, unconstrained optimization.
    # var = mu + mu^2 / r, so r -> inf recovers the Poisson (equidispersed) limit.
    def nll(theta):
        mu, log_r = theta
        if mu <= 0:
            return np.inf
        r = math.exp(log_r)
        p = r / (r + mu)
        return -stats.nbinom.logpmf(data, r, p).sum()

    r_seed = m * m / (v - m) if v > m else 1e4  # method-of-moments seed
    fit = minimize(nll, [m, math.log(max(r_seed, 1e-3))], method="Nelder-Mead")
    if np.isfinite(fit.fun):
        mu, r = fit.x[0], math.exp(fit.x[1])
        p = r / (r + mu)
        cand["NegBinom"] = {
            "aic": 2 * 2 + 2 * fit.fun,  # fit.fun is the negative log-likelihood
            "pmf": lambda k, r=r, p=p: stats.nbinom.pmf(k, r, p),
        }

    best = min(cand, key=lambda name: cand[name]["aic"])
    return best, cand


def plot_histo(stat: str, history: list, player_name: str, opp_team: str, date: str):

    bins = range(math.floor(min(history)), math.ceil(max(history)) + 2)
    counts, edges, _ = plt.hist(
        history,
        bins=bins,
        edgecolor="springgreen",
        facecolor="none",
        linewidth=1.5,
    )
    centers = (edges[:-1] + edges[1:]) / 2

    # --- data-driven line of best fit -------------------------------------
    # Fit the PMF at the integer outcomes, but PLOT at the bar centers
    # (value + 0.5) so the curve lines up with the width-1 histogram bars.
    values = np.asarray(bins[:-1])  # actual outcomes: 0, 1, 2, ...
    name, models = fit_best_count_model(history)
    y = models[name]["pmf"](values) * len(history)  # PMF -> expected frequency
    plt.plot(
        centers,
        y,
        color="deepskyblue",
        linewidth=2,
        label=f"{name} fit (AIC {models[name]['aic']:.0f})",
    )
    plt.legend(facecolor="black", edgecolor="white", labelcolor="white", fontsize=8)
    # ----------------------------------------------------------------------

    plt.xticks(centers, bins[:-1])
    plt.xlabel(stat)
    plt.ylabel("Frequency")
    plt.title(f"{player_name} {stat} Distribution vs {opp_team} {date}")

    plt.axvline(np.mean(history), color="tomato", linestyle="solid", linewidth=3)
    plt.axvline(
        np.mean(history) + np.std(history),
        color="tomato",
        linestyle=(0, (1, 4)),
        linewidth=3,
    )
    plt.axvline(
        np.mean(history) - np.std(history),
        color="tomato",
        linestyle=(0, (1, 4)),
        linewidth=3,
    )
    plt.show()
