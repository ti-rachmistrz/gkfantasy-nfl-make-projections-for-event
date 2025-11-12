# touchdowns_zinb_fit.py
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.optimize import minimize

def _tail_gt_nb(t: float, mu: float, r: float) -> float:
    """
    Tail for plain NB: P(X > t), mapping t (possibly non-integer) to discrete cutoff.
    scipy.nbinom with n=r, p=r/(r+mu).
    """
    if mu <= 0 or r <= 0:
        return np.nan
    n = r
    p = r / (r + mu)
    k = int(np.floor(t))
    if abs(t - round(t)) < 1e-12:
        k = int(round(t))
    return float(1.0 - nbinom.cdf(k, n, p))

def _tail_gt_zinb(t: float, mu: float, r: float, pi0: float) -> float:
    """
    Tail for ZINB mixture:
      With prob pi0: structural zero (Z=0)
      With prob 1-pi0: Y ~ NB(mu, r)
      X = 0 with prob pi0 + (1-pi0)*P_NB(0), and for k>0, P(X=k)=(1-pi0)P_NB(k).
    Therefore:
      P(X > t) = (1 - pi0) * P_NB(Y > t)
    """
    return (1.0 - pi0) * _tail_gt_nb(t, mu, r)

def implied_mean_zinb_from_tails(devig_df: pd.DataFrame) -> float:
    """
    Fit a Zero-Inflated Negative Binomial (ZINB) to de-vigged over/under probabilities for TDs.
    Returns the implied mean E[X] = (1 - pi0) * mu.

    Input (from devig_decimal_simple):
      columns:
        - 'overUnder' in {'over','under'}
        - 'point'      numeric threshold (often 0.5, 1.5, ...)
        - 'p_devig'    de-vigged probability of the listed side

    Fitting:
      For each i:
        s_i (target) = P(X > t_i)
          = p_devig           if over
          = 1 - p_devig       if under
        q_i(theta) = S_ZINB(t_i | mu, r, pi0)
      Minimize cross-entropy: sum_i [-s_i log q_i - (1-s_i) log(1-q_i)]
      Parameters: mu>0, r>0, pi0 in [0,1)

    Returns:
      float: implied mean (1 - pi0_hat) * mu_hat
    """
    req = {"overUnder", "point", "p_devig"}
    if not req.issubset(devig_df.columns):
        raise ValueError(f"devig_df must contain columns {sorted(req)}")

    df = devig_df.copy()
    if len(df) < 2:
        raise ValueError("Need at least two constraints to fit ZINB.")

    is_over = df["overUnder"].str.lower().values == "over"
    p = df["p_devig"].astype(float).values
    s_target = np.where(is_over, p, 1.0 - p)
    s_target = np.clip(s_target, 1e-9, 1 - 1e-9)
    tvals = df["point"].astype(float).values

    def loss(raw_params):
        # params: log_mu, log_r, logit_pi0
        log_mu, log_r, logit_pi0 = raw_params
        mu = np.exp(log_mu)
        r  = np.exp(log_r)
        # sigmoid but capped to avoid exactly 1
        pi0 = 1.0 / (1.0 + np.exp(-logit_pi0))
        pi0 = min(max(pi0, 1e-9), 1 - 1e-9)

        q = np.array([_tail_gt_zinb(t, mu, r, pi0) for t in tvals], dtype=float)
        q = np.clip(q, 1e-12, 1 - 1e-12)
        ce = -(s_target * np.log(q) + (1 - s_target) * np.log(1 - q))
        return float(np.sum(ce))

    # Initial guesses:
    # - start mu near median-ish threshold, r mild overdispersion, pi0 modest (lots of zeros for TDs)
    mu0   = max(0.3, np.median(np.maximum(0.0, tvals)) / 2.0)  # small starting rate
    r0    = 2.0
    pi00  = 0.4
    x0 = np.array([np.log(mu0), np.log(r0), np.log(pi00/(1-pi00))])

    res = minimize(loss, x0=x0, method="Nelder-Mead", options={"maxfev": 30000, "maxiter": 30000})
    log_mu_hat, log_r_hat, logit_pi0_hat = res.x
    mu_hat  = float(np.exp(log_mu_hat))
    r_hat   = float(np.exp(log_r_hat))
    pi0_hat = float(1.0 / (1.0 + np.exp(-logit_pi0_hat)))
    pi0_hat = min(max(pi0_hat, 1e-9), 1 - 1e-9)

    implied_mean = (1.0 - pi0_hat) * mu_hat
    return float(implied_mean)
