# receptions_nb_fit.py
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.optimize import minimize

def _tail_gt_nb(t: float, mu: float, r: float) -> float:
    """
    Model tail P(X > t) for a discrete NB variable when the sportsbook threshold t may be non-integer.
    NB parameterization: mean=mu (>0), dispersion r (>0). scipy.nbinom uses:
      n = r, p = r/(r+mu)
    Tail rule for discrete counts:
      if t is non-integer (e.g., 5.5): P(X > t) = P(X >= floor(t)+1) = 1 - CDF(floor(t))
      if t is integer (e.g., 5):       P(X > t) = 1 - CDF(t)
    """
    if mu <= 0 or r <= 0:
        return np.nan
    n = r
    p = r / (r + mu)
    k = int(np.floor(t))
    # if t is an integer within floating tolerance, we still want P(X > t) = 1 - CDF(t)
    if abs(t - round(t)) < 1e-12:
        k = int(round(t))
    return float(1.0 - nbinom.cdf(k, n, p))

def implied_mean_nb_from_tails(devig_df: pd.DataFrame) -> float:
    """
    Fit a Negative Binomial to de-vigged over/under probabilities for receptions and return implied mean (mu).

    Input (from devig_decimal_simple):
      columns:
        - 'overUnder' in {'over','under'}
        - 'point'      numeric threshold (can be .5 lines)
        - 'p_devig'    de-vigged probability of the listed side

    We fit (mu>0, r>0) by minimizing cross-entropy on tail probabilities:
      For each row i with threshold t_i:
        target s_i = P(X > t_i)
          = p_devig            if over
          = 1 - p_devig        if under
        model q_i = S_NB(t_i | mu, r)
      Loss = sum[ -s_i log q_i - (1-s_i) log(1-q_i) ]

    Returns:
      float: implied mean mu_hat
    """
    req = {"overUnder", "point", "p_devig"}
    if not req.issubset(devig_df.columns):
        raise ValueError(f"devig_df must contain columns {sorted(req)}")

    df = devig_df.copy()
    if len(df) < 2:
        raise ValueError("Need at least two constraints to fit NB.")

    # Build target tail probabilities s_i = P(X > t_i)
    is_over = df["overUnder"].str.lower().values == "over"
    p = df["p_devig"].astype(float).values
    s_target = np.where(is_over, p, 1.0 - p)
    s_target = np.clip(s_target, 1e-9, 1 - 1e-9)
    tvals = df["point"].astype(float).values

    def loss(log_params):
        mu = np.exp(log_params[0])        # >0
        r  = np.exp(log_params[1])        # >0
        q = np.array([_tail_gt_nb(t, mu, r) for t in tvals], dtype=float)
        q = np.clip(q, 1e-12, 1 - 1e-12)
        ce = -(s_target * np.log(q) + (1 - s_target) * np.log(1 - q))
        return float(np.sum(ce))

    # reasonable start: Poisson-ish with mild overdispersion
    x0 = np.log([max(1.0, np.median(tvals)), 2.0])
    res = minimize(loss, x0=x0, method="Nelder-Mead", options={"maxfev": 20000, "maxiter": 20000})
    mu_hat = float(np.exp(res.x[0]))
    return mu_hat
