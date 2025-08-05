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
    print(f"Player 2 card: ", end="")
    Card.print_pretty_cards(state[0])
    print(f"Player 1 action: ", end="")
    Card.print_pretty_cards(state[1])
    print(f"len_player1_card = {state[-1]}")

def process_computer(list_file):
    set_tmp = set()
    for file in tqdm(list_file):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            if len(data) == 1:
                state = list(data.keys())[0]
                set_tmp.add(state[-1])
                for action, Q_value in data[state].items():
                    assert -100 <= Q_value <= 100
            else:
                assert data["version"] == "final" 
                assert len(data) == 4
                assert data["who_play_first"] in [1, 2]
                assert len(data["player1_card"]) == 13
                assert len(data["player2_card"]) == 13
        except Exception as e:
            print("ERRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOORRRRRRRRRR: ", e)
            print(file)
    print(f"set_tmp = {set_tmp}")

if __name__ == "__main__":
    n_process = psutil.cpu_count(logical=False) - 6
    src_data = r"F:\TLMN_2_players\data_final_2_1"

    list_file = list(glob.glob(os.path.join(src_data, "**/*.pkl"), recursive=True))
    print(f"The number of file need to process: {len(list_file):,}")
    with Pool(processes=n_process) as pool:
        pool.map(process_computer, np.array_split(list_file, n_process))