import pandas as pd
import re
from typing import List, Dict, Any

class OddsApiResponse:
    def __init__(self, json_response: List[Dict[str, Any]]):
        self.json: List[Dict[str, Any]] = json_response
    

    def to_df(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        # loop through json, adding prop/lines as row
        for book in self.json.get("bookmakers", []):
            for market in book.get("markets", []):
                prop_key = market.get("key")
                for outcome in market.get("outcomes", []):
                    player_name = outcome.get("description")
                    over_under = (outcome.get("name") or "").strip().lower()
                    if over_under == "yes" and prop_key == "player_anytime_td":
                        over_under = "over"
                    point = outcome.get("point")
                    if point is None and prop_key == "player_anytime_td":
                        point = "0.5"
                    odds_val = outcome.get("price")

                    rows.append(
                        {
                            "player_name": player_name,
                            "player_name_normal": self._normalize_name(player_name),
                            "prop": prop_key,
                            "overUnder": over_under,
                            "point": point,
                            "odds": odds_val,
                        }
                    )
        return pd.DataFrame(rows, columns=["player_name", "player_name_normal", "prop", "overUnder", "point", "odds"])

    def _normalize_name(self, name: str) -> str:
        return re.sub(r'[^a-z]', '', name.lower())
