from dataclasses import dataclass, asdict

@dataclass
class Player:
    name: str
    proj_pass_yd: float = 0.0
    proj_rush_yd: float = 0.0
    proj_rec_yd: float = 0.0
    proj_pass_td: float = 0.0
    proj_rush_rec_td: float = 0.0
    proj_receptions: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
