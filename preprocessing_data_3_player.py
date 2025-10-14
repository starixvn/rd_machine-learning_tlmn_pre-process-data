import os
import pickle
import gc
import time
import polars as pl
import numpy as np
import psutil
from tienlen_libs import Utils
from multiprocessing import Process, Queue
from glob import glob
from itertools import chain
from tqdm import tqdm
from os.path import dirname, basename, join


def remove_folder_with_empty_state_file(data_src_dir):
    all_folder_data = glob(os.path.join(data_src_dir, '**'))
    for folder in all_folder_data:
        if os.path.isdir(folder):
            files_in_folder = os.listdir(folder)
            if len(files_in_folder) == 1 and 'initial_game_info.pkl' in files_in_folder:
                print(f"Removing folder and its contents: {folder}")
                for file in files_in_folder:
                    os.remove(os.path.join(folder, file))
                os.rmdir(folder)
        else:
            raise ValueError(f"Expected a directory but found a file: {folder}")

def remove_empty_file(data_src_dir):
    pkl_files = glob(os.path.join(data_src_dir, '**/state_*.pkl'), recursive=True)
    for file in pkl_files:
        if os.path.getsize(file) == 0:
            print(f"Found empty file: {file}")
            os.remove(file)

def cleaning_data(data_src_dir):
    print("\nCleaning data stage 1: Remove folder with empty state file")
    remove_folder_with_empty_state_file(data_src_dir)
    print("\nCleaning data stage 2: Remove empty file")
    remove_empty_file(data_src_dir)

def checking_duplicate_initial_game_info(data_src_dir):
    print(f"\nChecking duplicate initial game info:")
    all_initial_game_infos = glob(os.path.join(data_src_dir, '**/initial_game_info.pkl'), recursive=True)
    assert len(all_initial_game_infos) == len(os.listdir(data_src_dir))
    player_sets = [set() for _ in range(4)]
    for initial_game_info_dir in all_initial_game_infos:
        with open(initial_game_info_dir, 'rb') as f:
            initial_game_info = pickle.load(f)
        player_card = initial_game_info["player_card"]
        for i, cards in enumerate(player_card):
            player_sets[i].add(cards)
    for i, player_set in enumerate(player_sets):
        assert len(player_set) == len(all_initial_game_infos), f"Duplicate cards in {data_src_dir}"
        print(f"[TEST PASSED] The length of set_player{i}_card: {len(player_set)}")


def generate_one_hot_encoding(cards: tuple[int], dimension: int = 52):
    arr = np.zeros(dimension, dtype=bool)
    arr[list(cards)] = True
    return arr

def generate_action_encoding(action):
    action_encode = np.zeros(21 + 52, dtype=bool)

    if isinstance(action, int):
        assert action == -1 or action == -2
        action_int_combo_type = 18 - action
    else:
        action_int_combo_type = Utils.get_int_combo_type(action)
        assert action_int_combo_type < 19
    action_encode[action_int_combo_type] = True

    if not isinstance(action, int) and len(action) > 0:
        max_card = max(action)
        action_encode[21 + max_card] = True

    """Add color type of pair deuce if neccessary"""
    return action_encode

def generate_state_encoding(state, initial_player_card):
    n_player = len(initial_player_card)
    actions_card_player = [tuple(set(initial_player_card[i]) - set(state[i])) 
                            for i in range(n_player)]
    deck_remain = tuple(chain.from_iterable(state[1:n_player]))
    len_other_player_card = [len(state[i]) for i in range(1, n_player)]
    assert all(len(actions_card_player[i+1]) + len_other_player_card[i] == 13 
                for i in range(n_player-1))
    player_skip = np.asarray(state[2*n_player-1:3*n_player-1])
    turn = state[-1]

    return np.concatenate([
        generate_one_hot_encoding(state[0]),
        *(generate_action_encoding(state[i]) for i in range(n_player, 2*n_player-1)),
        *(generate_one_hot_encoding(cards) for cards in actions_card_player),
        generate_one_hot_encoding(deck_remain),
        *(generate_one_hot_encoding((l,), dimension=14) for l in len_other_player_card),
        player_skip,
        generate_one_hot_encoding((turn,), dimension=n_player),
    ])

def generate_list_state_action_return_encode(folder_data_dir):
    initial_game_info_dir = os.path.join(folder_data_dir, "initial_game_info.pkl")
    with open(initial_game_info_dir, "rb") as f:
        initial_game_info = pickle.load(f)
    assert initial_game_info["version"] == "medium_mode_fixed"
    initial_player_card = initial_game_info["player_card"]
    assert all(len(cards) == 13 for cards in initial_player_card)

    for file_data_dir in glob(os.path.join(folder_data_dir, "state_*.pkl")):
        with open(file_data_dir, "rb") as f:
            Q_table = pickle.load(f)

        for state, state_info in Q_table.items():
            assert len(state_info) > 1, f"len of Q_table is <= 1: {file_data_dir}"
            state_encoding = generate_state_encoding(state, initial_player_card)

            state_len = len(state_encoding)
            action_len = len(generate_action_encoding(-1))
            state_action_array = np.empty(state_len + action_len, dtype=bool)
            state_action_array[:state_len] = state_encoding

            for action, expected_return in state_info.items():
                assert -100 <= expected_return <= 100
                state_action_array[state_len:] = generate_action_encoding(action)

                yield state_action_array, expected_return

def build_X_schema_columns(n_player):
    schema_columns = []
    schema_columns.extend([f"our_card_{j}" for j in range(52)])
    for i in range(1, n_player):
        schema_columns.extend([f"player{i}_action_combo_type_{j}" for j in range(21)])
        schema_columns.extend([f"player{i}_action_max_card_{j}" for j in range(52)])
    for i in range(n_player):
        schema_columns.extend([f"actions_card_player{i}_{j}" for j in range(52)])
    schema_columns.extend([f"deck_remain_{i}" for i in range(52)])
    for i in range(1, n_player):
        schema_columns.extend([f"len_player{i}_card_{j}" for j in range(14)])
    schema_columns.extend([f"player_skip_{i}" for i in range(n_player)])
    schema_columns.extend([f"turn_{i}" for i in range(n_player)])
    schema_columns.extend([f"action_combo_type_{j}" for j in range(21)])
    schema_columns.extend([f"action_max_card_{j}" for j in range(52)])
    return schema_columns
X_schema_columns = build_X_schema_columns(n_player=4)

def create_polars_table_of_state_action_return_encode(tuple_folder_data_and_preprocess_dir):
    folder_data_dir, preprocess_file_dir = tuple_folder_data_and_preprocess_dir
    start_time = time.time()

    X, y = zip(*generate_list_state_action_return_encode(folder_data_dir))
    df_X = pl.DataFrame(np.stack(X), schema=X_schema_columns)
    df_y = pl.DataFrame({"expected_return": y})
    df_table = pl.concat([df_X, df_y], how="horizontal")
    df_table.write_parquet(preprocess_file_dir)
    print(f"Time to process folder {folder_data_dir} is: {time.time() - start_time} seconds")

def get_tuple_folder_data_and_preprocess_dir(data_src_dir, data_preprocess_folder_dir):
    for folder_data_name in os.listdir(data_src_dir):
        folder_data_dir = os.path.join(data_src_dir, folder_data_name)
        preprocess_file_dir = os.path.join(data_preprocess_folder_dir, folder_data_name) + ".parquet"
        if not os.path.exists(preprocess_file_dir):
            yield folder_data_dir, preprocess_file_dir

def worker(task_queue):
    while True:
        item = task_queue.get()
        if item is None:
            break
        try:
            create_polars_table_of_state_action_return_encode(item)
        except Exception as e:
            print(f"Error processing {item}: {e}", flush=True)

if __name__ == "__main__":
    n_process = psutil.cpu_count(logical=False) - 2
    task_queue = Queue()

    all_data_src_dir = r"C:\Users\justin.pham\Downloads\test_tmp"
    list_all_data_src_folder = glob(os.path.join(all_data_src_dir, '*'))
    print(f"list_all_data_src_folder: {list_all_data_src_folder}")

    workers = [Process(target=worker, args=(task_queue,)) for _ in range(n_process)]
    for w in workers:
        w.start()

    for data_src_dir in list_all_data_src_folder:
        print(f"\nProcessing the {data_src_dir} folder...........")
        cleaning_data(data_src_dir)
        checking_duplicate_initial_game_info(data_src_dir)
        data_preprocess_folder_dir = join(dirname(data_src_dir), "preprocess", 
                                        basename(data_src_dir) + "_preprocess")
        os.makedirs(data_preprocess_folder_dir)
        
        for item in get_tuple_folder_data_and_preprocess_dir(data_src_dir, data_preprocess_folder_dir):
            task_queue.put(item)

    for _ in range(n_process):
        task_queue.put(None)
        
    for w in workers:
        w.join()
    print("✅ All folders processed dynamically ❤️‍🔥")