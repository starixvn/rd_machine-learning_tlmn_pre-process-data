import polars as pl
from glob import glob
import os
from tqdm import tqdm
import pickle
import numpy as np
from multiprocessing import Pool
import psutil

def process_computer(list_file):
    len_player1_card_1 = 0
    len_player1_card_2 = 0
    for file in tqdm(list_file):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            assert len(data) == 1
            for state in data:
                len_player1_card = state[-1]
                assert len_player1_card in [1, 2]
                if len_player1_card == 1:
                    len_player1_card_1 += 1
                else:
                    len_player1_card_2 += 1
                break
        except BaseException as e:
            print("ERRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOORRRRRRRRRR")
    return len_player1_card_1, len_player1_card_2

if __name__ == "__main__":
    n_process = psutil.cpu_count(logical=False)
    src_data = r"D:\Project\TLMN\data_2_1"

    list_file = list(glob(os.path.join(src_data, "**/state_*.pkl"), recursive=True))
    print(f"The number of file need to process: {len(list_file):,}")
    with Pool(processes=n_process) as pool:
        tmp = pool.map(process_computer, np.array_split(list_file, n_process))
        len_player1_card_1 = sum(x[0] for x in tmp)
        len_player1_card_2 = sum(x[1] for x in tmp)
        print(f"\nNumber of states with len_player1_card = 1: {len_player1_card_1:,}")
        print(f"Number of states with len_player1_card = 2: {len_player1_card_2:,}")
        print(f"Total states: {len_player1_card_1 + len_player1_card_2:,}")