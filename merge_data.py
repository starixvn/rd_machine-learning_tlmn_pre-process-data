import polars as pl
import os
import glob
import gc

src_data = r"F:\dataset_2"
for folder_dataset_dir in glob.glob(os.path.join(src_data, '*')):
    output_file_dir = folder_dataset_dir + ".parquet"
    print(f"Merging the folder data {folder_dataset_dir}")
    df = pl.scan_parquet(os.path.join(folder_dataset_dir, "*.parquet"), low_memory=True)
    print(df)
    print(f"Shuffling the merge data ....................")
    df = df.collect().sample(fraction=1.0, shuffle=True)
    print(f"Shuffling DONE => Writing to {output_file_dir}")
    df.write_parquet(output_file_dir)
    print(f"Writing DONE")
    del df; gc.collect()