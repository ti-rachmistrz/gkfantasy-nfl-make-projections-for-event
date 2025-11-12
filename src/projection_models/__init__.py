
from .log_normal_fit import implied_mean_lognormal_from_tails
from .receptions_nb_fit import implied_mean_nb_from_tails
from .touchdowns_zinb_fit import implied_mean_zinb_from_tails

__all__ = [
    "implied_mean_zinb_from_tails", 
    "implied_mean_nb_from_tails", 
    "implied_mean_lognormal_from_tails"
]
