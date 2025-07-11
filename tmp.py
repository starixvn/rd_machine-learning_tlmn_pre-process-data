import polars as pl
from glob import glob
import os
from tqdm import tqdm
import pickle
import numpy as np
from multiprocessing import Pool
import psutil
from tienlen_libs import Card

def process_computer(list_file):
    for file in (list_file):
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
            assert len(data) == 1
            for state in data:
                if set(state[0]).issubset({48, 49, 50, 51}) and len(state[0]) >= 2 and state[1] == tuple():
                    Card.print_pretty_cards(state[0])
                    print(data, end="\n\n")
        except Exception as e:
            print("ERRRRRRRRRRRRRRRRRRRRROOOOOOOOOOOOORRRRRRRRRR")

if __name__ == "__main__":
    n_process = psutil.cpu_count(logical=False)
    src_data = r"D:\Project\TLMN\data_2_1"

    list_file = list(glob(os.path.join(src_data, "**/state_*.pkl"), recursive=True))
    print(f"The number of file need to process: {len(list_file):,}")
    with Pool(processes=n_process) as pool:
        pool.map(process_computer, np.array_split(list_file, n_process))