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
    set_len_player1_card = set()
    count = 0
    for file in list_file:
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            assert len(data) == 1
            state = list(data.keys())[0]
            if state[-1] == 13 and len(state[0]) == 13:
                count += 1
                print_state(state)
                for action in data[state]:
                    print(f"Action: ", end="")
                    Card.print_pretty_cards(action, end=" => ")
                    print(f"Q_value: {data[state][action]}")
        except Exception as e:
            print("ERRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOORRRRRRRRRR", e)
    print(f"The unique len_player1_card: {set_len_player1_card}")
    print(f"Total count: {count}")

if __name__ == "__main__":
    n_process = psutil.cpu_count(logical=False) - 11
    src_data = r"D:\Project\TLMN\2_players\data_full_13_11"

    list_file = list(glob.glob(os.path.join(src_data, "**/state_*.pkl"), recursive=True))
    print(f"The number of file need to process: {len(list_file):,}")
    with Pool(processes=n_process) as pool:
        pool.map(process_computer, np.array_split(list_file, n_process))