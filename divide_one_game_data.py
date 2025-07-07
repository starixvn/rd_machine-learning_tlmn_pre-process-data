import polars as pl
import os
import glob
import math
import psutil
from multiprocessing import Pool, freeze_support
import numpy as np
import sys
from tqdm import tqdm


def find_root_file_name(game_data_file):
    file_name = os.path.basename(game_data_file)
    assert file_name.endswith(".parquet"), "File name must end with .parquet"
    root_file_name, ext = os.path.splitext(file_name)
    return root_file_name

def check_data_after_divide(game_data_file, write_folder):
    src_data = pl.read_parquet(game_data_file)
    merge_data = pl.read_parquet(os.path.join(write_folder, "*.parquet"))
    assert src_data.shape == merge_data.shape, "Data after divide have different shapes"
    # Check that no file has length 0
    for file in glob.glob(os.path.join(write_folder, "*.parquet")):
        assert len(pl.read_parquet(file)) > 0

def divide_data_one_game_into_multiple_parts(game_data_file, write_dir, n_part):
    root_file_name = find_root_file_name(game_data_file)
    write_foler = os.path.join(write_dir, root_file_name)
    os.makedirs(write_foler, exist_ok=True)

    data = pl.read_parquet(game_data_file)
    len_data = len(data)
    data = data.sample(fraction=1.0, shuffle=True)
    n_sample_per_part = math.ceil(len_data / n_part)
    for part_idx, i in enumerate(range(0, len_data, n_sample_per_part)):
        subdata = data.slice(i, n_sample_per_part)
        subdata.write_parquet(os.path.join(write_foler, 
                                        f"{root_file_name}_part_{part_idx}.parquet"))
    check_data_after_divide(game_data_file, write_foler)

def process_divide_list_data_one_game(list_tuple_args):
    for tuple_args in tqdm(list_tuple_args):
        game_data_file, write_dir, n_part = tuple_args
        divide_data_one_game_into_multiple_parts(game_data_file, write_dir, int(n_part))

if __name__ == "__main__":
    def get_list_tuple_args(data_src_dir, write_dir, n_part):
        for file_data_name in os.listdir(data_src_dir):
            file_data_dir = os.path.join(data_src_dir, file_data_name)
            yield file_data_dir, write_dir, n_part

    n_process = psutil.cpu_count(logical=False)
    data_src_dir = r"F:\data_see_opponent_card_4_players\data_validation"
    write_dir = r"F:\data_see_opponent_card_4_players\data_validation_divide"
    n_part = 10
    
    print(f"\nProcessing the {data_src_dir} folder...........")
    os.makedirs(write_dir)
    list_tuple_args = list(get_list_tuple_args(data_src_dir, write_dir, n_part))
    print(f"The number of file need to process: {len(list_tuple_args):,}, TOTAL: {len(os.listdir(data_src_dir)):,}")
    if sys.platform.startswith('win'):
        freeze_support()
    with Pool(processes=n_process) as pool:
        pool.map(process_divide_list_data_one_game, 
                np.array_split(list_tuple_args, n_process))