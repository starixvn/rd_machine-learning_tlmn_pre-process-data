import os
import pickle
import gc
from tqdm import tqdm
import time
import polars as pl
import numpy as np
import psutil
import sys
import glob
from scipy import sparse
from tienlen_libs import Utils, ActionGenerator, Card
from os.path import dirname, basename, join
from multiprocessing import Pool, freeze_support
import math
from collections import defaultdict


class FeatureEngineering(object):
    def __init__(self):
        self.mapping_suit_int_to_str = {0: "spade", 1: "club", 2: "diamond", 3: "heart"}
        self.__check_mapping_suit_int_to_str()
        self.mapping_rank_int_to_str = {0: "trey", 1: "four", 2: "five", 3: "six", 4: "seven", 5: "eight",
                            6: "nine", 7: "ten", 8: "jack", 9: "queen", 10: "king", 11: "ace", 12: "deuce"}
        self.list_key_cards_info = self.__get_list_key_cards_info()
        self.action_generator = ActionGenerator()
        self.utils = Utils()
        self.list_key_n_sequence_double_sequence_info = self.__get_list_key_n_sequence_double_sequence_info()
        
    def __check_mapping_suit_int_to_str(self):
        for suit_int in self.mapping_suit_int_to_str:
            assert self.mapping_suit_int_to_str[suit_int][0] == Card.INT_SUIT_TO_CHAR_SUIT[suit_int]

    def __get_list_key_cards_info(self):
        list_key = []
        for rank_str in self.mapping_rank_int_to_str.values():
            list_key.append(f"n_{rank_str}")
        for suit_str in self.mapping_suit_int_to_str.values():
            list_key.append(f"n_{suit_str}")
        assert len(list_key) == 13 + 4, f"The length of list_key is not correct: {len(list_key)}"
        return list_key
    
    def __get_list_key_n_sequence_double_sequence_info(self):
        list_key = []
        for i in range(3, 13):
            list_key.append(f"n_sequence_{i}")
        for i in range(3, 5):
            list_key.append(f"n_double_sequence_{i}")
        assert len(list_key) == 10 + 2, f"The length of list_key is not correct: {len(list_key)}"
        return list_key
    
    def stats_cards_info(self, cards: tuple[int]):
        cards_info = defaultdict(int)
        for card_int in cards:
            rank_str = self.mapping_rank_int_to_str[Card.get_rank_int(card_int)]
            suit_str = self.mapping_suit_int_to_str[Card.get_suit_int(card_int)]

            cards_info[f"n_{rank_str}"] += 1
            cards_info[f"n_{suit_str}"] += 1

        assert set(cards_info.keys()).issubset(set(self.list_key_cards_info))
        return tuple(cards_info[key] for key in self.list_key_cards_info)
    
    def __is_double_sequence(self, rank_sequence_n_card: list[int]):
        for rank_n_card in rank_sequence_n_card:
            if rank_n_card < 2:
                return False
        return True

    def __count_n_double_sequence(self, rank_sequence_n_card: list[int]):
        return np.prod([math.comb(n_card, 2) for n_card in rank_sequence_n_card])
    
    def stats_n_sequence_double_sequence(self, cards: tuple[int]):
        frequency_stats = self.utils.stats_frequency_of_card(cards)
        list_rank = [rank for rank in frequency_stats.keys() if rank != 12]

        n_sequence_double_sequence_info = defaultdict(int)
        for rank_sequence in self.action_generator.detect_sequence_based_on_rank(list_rank):
            sequence_length = len(rank_sequence)
            rank_sequence_n_card = [len(frequency_stats[rank]) for rank in rank_sequence]
            n_sequence_double_sequence_info[f"n_sequence_{sequence_length}"] += np.prod(rank_sequence_n_card)
            
            if sequence_length <= 4 and self.__is_double_sequence(rank_sequence_n_card):
                n_double_sequence = self.__count_n_double_sequence(rank_sequence_n_card)
                n_sequence_double_sequence_info[f"n_double_sequence_{sequence_length}"] += n_double_sequence

        assert set(n_sequence_double_sequence_info.keys()).issubset(set(self.list_key_n_sequence_double_sequence_info))
        return tuple(n_sequence_double_sequence_info[key] for key in self.list_key_n_sequence_double_sequence_info)


def remove_folder_with_empty_state_file(data_src_dir):
    all_folder_data = glob.glob(os.path.join(data_src_dir, '**'))
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
    pkl_files = glob.glob(os.path.join(data_src_dir, '**/state_*.pkl'), recursive=True)
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
    all_initial_game_infos = glob.glob(os.path.join(data_src_dir, '**/initial_game_info.pkl'), recursive=True)
    assert len(all_initial_game_infos) == len(os.listdir(data_src_dir))
    set_player1_card = set()
    set_player2_card = set()
    for initial_game_info_dir in all_initial_game_infos:
        with open(initial_game_info_dir, 'rb') as f:
            initial_game_info = pickle.load(f)
        set_player1_card.add(initial_game_info["player1_card"])
        set_player2_card.add(initial_game_info["player2_card"])
    assert len(set_player1_card) == len(all_initial_game_infos)
    assert len(set_player2_card) == len(all_initial_game_infos)
    print(f"[TEST PASSED] The length of set_player1_card: {len(set_player1_card)}")
    print(f"[TEST PASSED] The length of set_player2_card: {len(set_player2_card)}")

def generate_one_hot_encoding(cards: tuple[int], dimension: int = 52):
    return tuple(True if i in cards else False for i in range(dimension))

def generate_action_encoding(action: tuple[int]):
    action_int_combo_type = Utils.get_int_combo_type(action)
    action_int_combo_type_encoding = generate_one_hot_encoding((action_int_combo_type,),
                                                               dimension=19)
    assert action_int_combo_type_encoding.count(True) == 1        
    if len(action) > 0:
        max_card = max(action)
        max_card_encoding = generate_one_hot_encoding((max_card,))
        assert max_card_encoding.count(True) == 1
    else:
        max_card_encoding = tuple(False for _ in range(52))
    return action_int_combo_type_encoding + max_card_encoding

def generate_list_state_action_return_encode(folder_data_dir):
    initial_game_info_dir = os.path.join(folder_data_dir, "initial_game_info.pkl")
    with open(initial_game_info_dir, "rb") as f:
        initial_game_info = pickle.load(f)
    player2_play_first = True if initial_game_info["who_play_first"] == 2 else False
    list_file_data_dir = glob.glob(os.path.join(folder_data_dir, "state_*.pkl"))
    assert len(list_file_data_dir) == len(os.listdir(folder_data_dir)) - 1
    for file_data_dir in list_file_data_dir:
        with open(file_data_dir, "rb") as f:
            Q_table = pickle.load(f)

        for state in Q_table:
            if len(Q_table[state]) <= 1: continue
            player2_card, player1_action, actions_card_player1, actions_card_player2, deck_remain, len_player1_card = state

            player2_card_encoding = generate_one_hot_encoding(player2_card)
            player1_action_encoding = generate_action_encoding(player1_action)
            actions_card_player1_encoding = generate_one_hot_encoding(actions_card_player1)
            actions_card_player2_encoding = generate_one_hot_encoding(actions_card_player2)
            deck_remain_encoding = generate_one_hot_encoding(deck_remain)
            len_player1_card_encoding = generate_one_hot_encoding((len_player1_card-1,), dimension=13)

            for action in Q_table[state]:
                expected_return = Q_table[state][action]
                assert 0 <= expected_return <= 1

                action_encoding = generate_action_encoding(action)

                state_action_return_encode = player2_card_encoding + player1_action_encoding + \
                                            actions_card_player1_encoding + actions_card_player2_encoding + \
                                            deck_remain_encoding + len_player1_card_encoding + \
                                            action_encoding + (player2_play_first, expected_return)
                yield state_action_return_encode

def create_polars_table_of_state_action_return_encode(list_tuple_folder_data_and_preprocess_dir):
    schema_columns = [(f"player2_card_{i}", pl.Boolean) for i in range(52)] + \
                    [(f"player1_action_combo_type_{i}", pl.Boolean) for i in range(19)] + \
                    [(f"player1_action_max_card_{i}", pl.Boolean) for i in range(52)] + \
                    [(f"actions_card_player1_{i}", pl.Boolean) for i in range(52)] + \
                    [(f"actions_card_player2_{i}", pl.Boolean) for i in range(52)] + \
                    [(f"deck_remain_{i}", pl.Boolean) for i in range(52)] + \
                    [(f"len_player1_card_{i+1}", pl.Boolean) for i in range(13)] + \
                    [(f"action_combo_type_{i}", pl.Boolean) for i in range(19)] + \
                    [(f"action_max_card_{i}", pl.Boolean) for i in range(52)] + \
                    [(f"player2_play_first", pl.Boolean)] + \
                    [(f"expected_return", pl.Float64)]
    for folder_data_dir, preprocess_file_dir in tqdm(list_tuple_folder_data_and_preprocess_dir):
        start_time = time.time()
        df_table = pl.DataFrame(generate_list_state_action_return_encode(folder_data_dir), 
                                schema=schema_columns)
        df_table.write_parquet(preprocess_file_dir)
        # print(f"Time to process folder {folder_data_dir} is: {time.time() - start_time} seconds")
        del df_table; gc.collect()

if __name__ == "__main__":
    n_process = psutil.cpu_count(logical=False)
    data_src_dir = r"D:\Project\TLMN\data_13_11"

    cleaning_data(data_src_dir)
    checking_duplicate_initial_game_info(data_src_dir)
    def get_tuple_folder_data_and_preprocess_dir(data_src_dir, data_preprocess_folder_dir):
        for folder_data_name in os.listdir(data_src_dir):
            folder_data_dir = os.path.join(data_src_dir, folder_data_name)
            preprocess_file_dir = os.path.join(data_preprocess_folder_dir, folder_data_name) + ".parquet"
            if not os.path.exists(preprocess_file_dir):
                yield folder_data_dir, preprocess_file_dir
    data_preprocess_folder_dir = join(dirname(data_src_dir), "preprocess", 
                                      basename(data_src_dir) + "_preprocess")
    os.makedirs(data_preprocess_folder_dir, exist_ok=True)
    print(f"\nProcessing the {data_src_dir} folder...........")
    list_tuple_folder_data_and_preprocess_dir = \
        list(get_tuple_folder_data_and_preprocess_dir(data_src_dir, data_preprocess_folder_dir))
    print(f"The number of folder need to process: {len(list_tuple_folder_data_and_preprocess_dir)}", end=", ")
    print(f"TOTAL: {len(os.listdir(data_src_dir))}")
    if sys.platform.startswith('win'):
        freeze_support()
    with Pool(processes=n_process) as pool:
        pool.map(create_polars_table_of_state_action_return_encode, 
                np.array_split(list_tuple_folder_data_and_preprocess_dir, n_process))