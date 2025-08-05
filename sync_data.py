import glob
import os
import time
from tqdm import tqdm
import shutil

def copy_data(src_data, des_dir):
    for entry in os.scandir(src_data):
        assert entry.is_dir(follow_symlinks=False)
        print(entry.name)
        des_entry = os.path.join(des_dir, entry.name)
        os.makedirs(des_entry, exist_ok=True)
        for sub_entry in os.scandir(entry.path):
            assert sub_entry.is_file()
            des_sub_entry = os.path.join(des_entry, sub_entry.name)
            if not os.path.exists(des_sub_entry):
                shutil.copy(sub_entry.path, des_sub_entry)

def check_data(src_data, des_dir):
    print(f"Checking the data {src_data} ........ ")
    for entry in tqdm(os.scandir(src_data)):
        assert entry.is_dir(follow_symlinks=False)
        des_entry = os.path.join(des_dir, entry.name)
        os.makedirs(des_entry, exist_ok=True)
        for sub_entry in os.scandir(entry.path):
            assert sub_entry.is_file()
            des_sub_entry = os.path.join(des_entry, sub_entry.name)
            assert os.path.exists(des_sub_entry), \
                f"{des_sub_entry} is not existing"
            assert sub_entry.stat().st_size == os.stat(des_sub_entry).st_size, \
                f"Size mismatch between {sub_entry.path} and {des_sub_entry}"
    print("PASSED")


start_time = time.time()

# copy_data(r"\\10.18.200.241\public\aibot\data_full_13_11", r"F:\TLMN_2_players\data_final_13_11")
# copy_data(r"\\10.18.200.241\public\aibot\data_full_10_8", r"F:\TLMN_2_players\data_final_10_8")
# copy_data(r"\\10.18.200.241\public\aibot\data_full_7_5", r"F:\TLMN_2_players\data_final_7_5")

check_data(r"\\10.18.200.241\public\aibot\data_full_13_11", r"F:\TLMN_2_players\data_final_13_11")
check_data(r"\\10.18.200.241\public\aibot\data_full_10_8", r"F:\TLMN_2_players\data_final_10_8")
check_data(r"\\10.18.200.241\public\aibot\data_full_7_5", r"F:\TLMN_2_players\data_final_7_5")

print(f"Time taken: {time.time() - start_time:.2f} seconds")