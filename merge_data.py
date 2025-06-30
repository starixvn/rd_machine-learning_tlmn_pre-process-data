import more_itertools
import glob
import os
import polars as pl
import gc

def yield_batch_interleave_data(src_data, batch_size):
    args = tuple(tuple(glob.glob(os.path.join(subfolder_dir, '*'))) 
                for subfolder_dir in glob.glob(os.path.join(src_data, '*')))
    interleave_data = more_itertools.interleave_longest(*args)
    for batch_data in more_itertools.batched(interleave_data, batch_size):
        yield batch_data

src_data = r"D:\Project\TLMN\data_see_opponent_card_3_players_divide\data_train"
batch_size = 23500
for idx, batch_data in enumerate(yield_batch_interleave_data(src_data, batch_size)):
    print(f"\nThe number of batch_data is {len(batch_data):,}")
    print(f"5 element first: ")
    for elem in batch_data[:5]: print(elem)
    print(f"5 element last: ")
    for elem in batch_data[-5:]: print(elem)
    if idx <= 33: continue
    df = pl.read_parquet(batch_data, low_memory=True)
    print(df.head())
    print(f"The shape of the df is: ({df.shape[0]:,}, {df.shape[1]})")
    print(f"Shuffling the merge data ....................")
    df = df.sample(fraction=1.0, shuffle=True)
    # output_file_dir = src_data + f"_{idx}.parquet"
    output_file_dir = fr"F:\data_see_opponent_card_3_players_divide\data_train\data_train_{idx}.parquet"
    print(f"Shuffling DONE => Writing to {output_file_dir}")
    df.write_parquet(output_file_dir)
    print(f"Writing DONE")
    del df; gc.collect()