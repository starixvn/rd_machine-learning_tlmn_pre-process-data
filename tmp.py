import polars as pl
from glob import glob
import os
from tqdm import tqdm
import pickle
import numpy as np
from multiprocessing import Pool
import psutil
from tienlen_libs import Card

def print_state(state):
    print(f"Player 2 card: ", end="")
    Card.print_pretty_cards(state[0])
    print(f"Player 1 action: ", end="")
    Card.print_pretty_cards(state[1])
    print(f"len_player1_card = {state[-1]}")

def process_computer(list_file):
    for file in (list_file):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            assert len(data) == 1
            state = list(data.keys())[0]
            if state[-1] == 10:
                print()
                print_state(state)
                for action in data[state]:
                    print(f"Action: ", end="")
                    Card.print_pretty_cards(action, end="; ")
                    print(f"Expected Reward: {data[state][action]}")
        except Exception as e:
            print("ERRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOORRRRRRRRRR")

if __name__ == "__main__":
    n_process = psutil.cpu_count(logical=False)
    src_data = r"D:\Project\TLMN\2_players\data_full_2_1"

    list_file = list(glob(os.path.join(src_data, "**/state_*.pkl"), recursive=True))
    print(f"The number of file need to process: {len(list_file):,}")
    with Pool(processes=n_process) as pool:
        pool.map(process_computer, np.array_split(list_file, n_process))