import numpy as np
import pandas as pd

def devig_decimal(df: pd.DataFrame, default_overround: float = 1.05) -> pd.DataFrame:
    """
    De-vig decimal odds for over/under markets.

    Rules:
    - If a point has BOTH 'over' and 'under', de-vig by normalizing within the pair
      so p(over)+p(under)=1 at that point.
    - If a point is single-sided (only over OR only under), de-vig by dividing raw prob
      by the fixed default_overround

    Parameters
    ----------
    df : DataFrame with columns: ['overUnder', 'point', 'odds']
         - overUnder ∈ {'over','under'}
         - odds are decimal (> 1.0)
    default_overround : float, default 1.05
         Assumed market overround for single-sided lines.

    Returns
    -------
    DataFrame: original columns plus
      - p_raw        : 1 / odds
      - overround_pair : sum of raw probs at that point if paired else NaN
      - p_devig      : de-vigged probability for the stated outcome
    """
    out = df.copy()
    if not {'overUnder','point','odds'} <= set(out.columns):
        raise ValueError("df must have columns: overUnder, point, odds")
    if (out['odds'] <= 1).any():
        raise ValueError("Decimal odds must be > 1.0")

    # Raw implied prob
    out['p_raw'] = 1.0 / out['odds']

    # Compute per-point overround ONLY when both sides exist
    g = out.groupby('point', group_keys=False)
    has_both = g['overUnder'].transform('nunique') == 2
    out['overround_pair'] = np.where(has_both, g['p_raw'].transform('sum'), np.nan)

    # De-vig
    eps = 1e-9
    out['p_devig'] = np.where(
        np.isfinite(out['overround_pair']),
        out['p_raw'] / out['overround_pair'],
        out['p_raw'] / float(default_overround)
    )
    out['p_devig'] = np.clip(out['p_devig'], eps, 1 - eps)

    return out
