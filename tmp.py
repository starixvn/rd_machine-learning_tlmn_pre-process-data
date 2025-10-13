import polars as pl
import glob
import os
from tqdm import tqdm
import pickle
import numpy as np
from multiprocessing import Pool
import psutil
from tienlen_libs import Card
from os.path import dirname, basename, join

def print_state(state):
    print(f"state = {state}")
    print(f"Player 2 card: ", end="")
    Card.print_pretty_cards(state[0])
    print(f"Player 1 action: ", end="")
    Card.print_pretty_cards(state[1])
    print(f"len_player1_card = {state[-1]}")

def process_computer(list_file):
    set_tmp = set()
    for file in tqdm(list_file):
        if os.path.getsize(file) == 0:
            continue
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            if len(data) == 1:
                state = list(data.keys())[0]
                set_tmp.add(state[-1])
                for action, Q_value in data[state].items():
                    assert -100 <= Q_value <= 100
                
                initial_game_info_dir = os.path.join(os.path.dirname(file), "initial_game_info.pkl")
                with open(initial_game_info_dir, "rb") as f:
                    initial_game_info = pickle.load(f)
                original_player2_card = initial_game_info["player2_card"]
                original_player1_card = initial_game_info["player1_card"]
                player2_card, player1_action, actions_card_player1, actions_card_player2, deck_remain, len_player1_card = state
                assert set(player2_card + actions_card_player2) == set(original_player2_card)
                assert set(player1_action).issubset(set(actions_card_player1))
                assert set(actions_card_player1).issubset(set(original_player1_card))
                assert len(actions_card_player1) + len_player1_card == 13
            else:
                assert data["version"] == "hard_mode" 
                assert len(data) == 4
                assert data["who_play_first"] in [1, 2]
                assert len(data["player1_card"]) == 13
                assert len(data["player2_card"]) == 13
        except Exception as e:
            print("ERRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOORRRRRRRRRR: ", e)
            print(file)
    print(f"set_tmp = {set_tmp}")

if __name__ == "__main__":
    n_process = psutil.cpu_count(logical=False) - 1
    src_data = r"F:\TLMN_2_players_hard_level\data_hard_2_1"

    list_file = list(glob.glob(os.path.join(src_data, "**/state_*.pkl"), recursive=True))
    print(f"The number of file need to process: {len(list_file):,}")
    with Pool(processes=n_process) as pool:
        pool.map(process_computer, np.array_split(list_file, n_process))