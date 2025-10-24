import glob
import os
import time
from tqdm import tqdm
import shutil

def copy_data(src_data, des_dir):
    command = fr"robocopy {src_data} {des_dir} /E /XC /MT:8 /R:3 /W:5 /Z /NFL"
    os.system(command)
    print(f"SUCCESSFULLY COPY folder {src_data} to {des_dir}")

def check_data(src_data, des_dir):
    print(f"Checking the data {src_data} ........ ")
    for entry in tqdm(os.scandir(src_data)):
        assert entry.is_dir(follow_symlinks=False)
        des_entry = os.path.join(des_dir, entry.name)
        for sub_entry in os.scandir(entry.path):
            assert sub_entry.is_file()
            des_sub_entry = os.path.join(des_entry, sub_entry.name)
            assert os.path.exists(des_sub_entry), \
                f"{des_sub_entry} is not existing"
            assert sub_entry.stat().st_size == os.stat(des_sub_entry).st_size, \
                f"Size mismatch between {sub_entry.path} and {des_sub_entry}"
    print("PASSED")


start_time = time.time()

# copy_data(r"\\10.18.200.241\public\aibot\data_full_13_11", r"F:\TLMN_4_players\medium\data_medium_13_11_add")
# copy_data(r"\\10.18.200.241\public\aibot\data_full_10_8", r"F:\TLMN_4_players\medium\data_medium_10_8_add")
# copy_data(r"\\10.18.200.241\public\aibot\data_full_7_5", r"F:\TLMN_4_players\medium\data_medium_7_5_add")
# copy_data(r"\\10.18.200.241\public\aibot\data_full_4_3", r"F:\TLMN_2_players_hard_level\data_hard_4_3")
# copy_data(r"\\10.18.200.241\public\aibot\data_full_2_1", r"F:\TLMN_2_players_hard_level\data_hard_2_1")

check_data(r"\\10.18.200.241\public\aibot\data_full_13_11", r"F:\TLMN_4_players\medium\data_medium_13_11")
check_data(r"\\10.18.200.241\public\aibot\data_full_10_8", r"F:\TLMN_4_players\medium\data_medium_10_8")
check_data(r"\\10.18.200.241\public\aibot\data_full_7_5", r"F:\TLMN_4_players\medium\data_medium_7_5")
# check_data(r"\\10.18.200.241\public\aibot\data_full_4_3", r"F:\TLMN_2_players_hard_level\data_hard_4_3")
# check_data(r"\\10.18.200.241\public\aibot\data_full_2_1", r"F:\TLMN_2_players_hard_level\data_hard_2_1")

print(f"Time taken: {time.time() - start_time:.2f} seconds")