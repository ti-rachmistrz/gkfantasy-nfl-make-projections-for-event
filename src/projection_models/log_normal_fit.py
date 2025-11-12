# lognormal_fit.py
import numpy as np
import pandas as pd
from scipy.stats import norm

def implied_mean_lognormal_from_tails(devig_df: pd.DataFrame) -> float:
    """
    Fit a Lognormal distribution to over/under probabilities and return the implied mean.

    Assumptions
    -----------
    - Input is the output of devig_decimal_simple(...), i.e., it includes columns:
        ['overUnder', 'point', 'p_devig']
      where:
        * 'overUnder' ∈ {'over','under'}
        * 'point' is the yardage threshold (> 0)
        * 'p_devig' is the de-vigged probability for that side (P(over) or P(under))

    Method
    ------
    If X ~ Lognormal(μ, σ^2) (i.e., ln X ~ Normal(μ, σ^2)), then the survival:
        S(t) = P(X > t) = Φ((μ - ln t) / σ)
    Taking probit on both sides:
        Φ^{-1}(S(t)) = (μ - ln t) / σ
    Rearranged:
        ln t = μ - σ * z,  where z := Φ^{-1}(S(t))
    So we do a simple least-squares regression of ln t on (-z) with an intercept μ.

    Returns
    -------
    float
        Implied mean of the fitted lognormal: exp(μ + 0.5 * σ^2)

    Raises
    ------
    ValueError
        If there are fewer than two valid constraints or all thresholds are invalid.
    """
    required_cols = {"overUnder", "point", "p_devig"}
    if not required_cols.issubset(set(devig_df.columns)):
        raise ValueError(f"devig_df must contain columns: {sorted(required_cols)}")

    # Coerce to numeric floats and sanitize
    df = devig_df.loc[:, ["overUnder", "point", "p_devig"]].copy()
    df["point"] = pd.to_numeric(df["point"])
    df["p_devig"] = pd.to_numeric(df["p_devig"])

    # Use only positive thresholds (lognormal support is x>0)
    df = df[df["point"] > 0].copy()
    if df.empty:
        raise ValueError("No positive thresholds available to fit a lognormal.")

    # Build target survival S(t) from either side's probability
    # - if row is 'over':     S = P(X>t) = p_devig
    # - if row is 'under':    S = P(X>t) = 1 - p_devig
    S = np.where(df["overUnder"].str.lower().values == "over",
                 df["p_devig"].values,
                 1.0 - df["p_devig"].values)

    # Numerical guards: clip to (0,1) to avoid ±inf probits
    eps = 1e-9
    S = np.clip(S, eps, 1 - eps)

    # Probit transform
    z = norm.ppf(S)

    # Regress ln t on (-z) with intercept:
    #   ln t_i = μ + β * x_i, with x_i = (-z_i) ; expect β ≈ σ  (since ln t = μ - σ z)
    ln_t = np.log(df["point"].values)
    X = np.c_[np.ones_like(z), -z]  # columns: [1, -z]

    # Need at least two distinct rows to estimate μ and σ
    if X.shape[0] < 2 or np.allclose(ln_t, ln_t[0]):
        if (len(devig_df) == 2 and len(devig_df["point"].unique()) == 1):
            return devig_df["point"].unique()[0]
        raise ValueError("Insufficient variation to fit lognormal (need ≥2 distinct constraints).")

    # Ordinary least squares
    beta, *_ = np.linalg.lstsq(X, ln_t, rcond=None)
    mu_hat, sigma_hat = float(beta[0]), float(beta[1])

    # σ must be positive; small negative can happen due to noise — take absolute value
    sigma_hat = abs(sigma_hat)
    if sigma_hat < 1e-12:
        # Degenerate: essentially a point mass; nudge to tiny variance
        sigma_hat = 1e-6

    # Implied mean of lognormal
    implied_mean = float(np.exp(mu_hat + 0.5 * sigma_hat**2))
    return implied_mean
