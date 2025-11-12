import os
import json

import pandas as pd

from player import Player
from s3_helper import S3Helper
from devig import devig_decimal
from projection_models import implied_mean_lognormal_from_tails, implied_mean_nb_from_tails, implied_mean_zinb_from_tails
from odds_api import OddsApiService, OddsApiResponse


def lambda_handler(event, context):
    s3_helper = S3Helper(bucket = 'gkfantasy-nfl')
    odds_api_service = OddsApiService(base_url = os.environ.get("THE_ODDS_API_BASE_URL"), secret_id = os.environ.get("SECRET_ID"))
    event_id = event['eventId']

    odds_api_response: OddsApiResponse = odds_api_service.get_event_odds(
        event_id = event_id, 
        markets = os.environ.get("MARKETS"), 
        bookmakers = os.environ.get("BOOKMAKERS")
    )

    lines_df = odds_api_response.to_df()

    players_list: list[Player] = get_players(lines_df)

    for p in players_list:
        player_lines_df = lines_df[lines_df["player_name_normal"] == p.name]
        # update_player_projections(p, player_df)
        p.proj_pass_yd = get_yardage(
            devig_decimal(player_lines_df[player_lines_df["prop"].isin(["player_pass_yds", "player_pass_yds_alternate"])])
        )
        p.proj_rush_yd = get_yardage(
            devig_decimal(player_lines_df[player_lines_df["prop"].isin(["player_rush_yds", "player_rush_yds_alternate"])])
        )
        p.proj_rec_yd = get_yardage(
            devig_decimal(player_lines_df[player_lines_df["prop"].isin(["player_reception_yds", "player_reception_yds_alternate"])])
        )
        p.proj_pass_td = get_touchdowns(
            devig_decimal(player_lines_df[player_lines_df["prop"].isin(["player_pass_tds", "player_pass_tds_alternate"])])
        )
        p.proj_rush_rec_td = get_touchdowns(
            devig_decimal(player_lines_df[player_lines_df["prop"].isin(["player_tds_over"])])
        )
        p.proj_receptions = get_receptions(
            devig_decimal(player_lines_df[player_lines_df["prop"].isin(["player_receptions", "player_receptions_alternate"])])
        )

    s3_helper.put(key = f"projections/{event_id}.json", content = json.dumps([p.to_dict() for p in players_list]))

    return {
        'statusCode': 200,
        'df': json.dumps([p.to_dict() for p in players_list])
    }



def get_players(df: pd.DataFrame) -> list[Player]:
    players = list()
    for player_name_normal in df["player_name_normal"].unique():
        players.append(Player(player_name_normal))
    return players

def get_yardage(df: pd.DataFrame) -> float:
    proj = 0
    if (df.empty):
        print("Empty dataframe passed to get_yardage(), skipping and returning 0...")
    else:
        try:
            proj = implied_mean_lognormal_from_tails(df)
        except ValueError as e:
            print(f"Error getting {df['prop'].unique()} for {df['player_name_normal'].unique()}.")
            print(df)
            print(e)
    return proj
    
def get_receptions(df: pd.DataFrame) -> float:
    proj = 0
    if (df.empty):
        print("Empty dataframe passed to get_receptions(), skipping and returning 0...")
    else:
        try:
            proj = implied_mean_nb_from_tails(df)
        except ValueError as e:
            print(f"Error getting {df['prop'].unique()} for {df['player_name_normal'].unique()}.")
            print(df)
            print(e)
    return proj
    
def get_touchdowns(df: pd.DataFrame) -> float:
    proj = 0
    if (df.empty):
        print("Empty dataframe passed to get_touchdowns(), skipping and returning 0...")
    else:
        try:
            proj = implied_mean_zinb_from_tails(df)
        except ValueError as e:
            print(f"Error getting {df['prop'].unique()} for {df['player_name_normal'].unique()}.")
            print(df)
            print(e)
    return proj
