import polars as pl
import os
import gc
import math
import keras
from keras import models, layers
import sys
import multiprocessing as mp
from glob import glob
from itertools import chain


class CustomDataGenerator(keras.utils.PyDataset):
    def __init__(self,
                list_src_data,
                dataset_type,
                n_worker,
                use_multiprocessing,
                max_queue_size,
                **kwargs):
        super().__init__(**kwargs)
        self.list_src_data = list_src_data
        assert dataset_type in ["train", "validation", "test"]
        self.dataset_type = dataset_type
        self.workers = n_worker
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size

        self.file_index = self.__build_file_index()
        print(f"The len of self.file_index is: {len(self.file_index):,}")
        self.bool_cols = self.__get_boolean_columns()

    def __build_file_index(self):
        return tuple(chain.from_iterable(
            glob(os.path.join(src_data_dir, '*'))
            for src_data_dir in self.list_src_data
        ))

    def __get_boolean_columns(self):
        sample_data_dir = self.file_index[0]
        schema = pl.scan_parquet(sample_data_dir).collect_schema()
        bool_cols = [col for col, dtype in schema.items() if dtype == pl.Boolean]
        assert len(bool_cols) == len(schema) - 4
        return bool_cols

    def __len__(self):
        return len(self.file_index)

    def normalize_dataset(self, dataset: pl.DataFrame):
        dataset = dataset.with_columns(
            [
                pl.col(self.bool_cols).cast(pl.Float64),  # Convert boolean columns to Float64
                ((pl.col("player2_action_max_card") + 1) / 52).cast(pl.Float64),  # Normalize player2_action_max_card
                ((pl.col("player3_action_max_card") + 1) / 52).cast(pl.Float64),  # Normalize player3_action_max_card
                ((pl.col("action_max_card") + 1) / 52).cast(pl.Float64),  # Normalize action_max_card
            ]
        )
        return dataset

    def __getitem__(self, index):
        batch_data = pl.read_parquet(self.file_index[index])
        batch_data = self.normalize_dataset(batch_data)
        batch_data = batch_data.to_numpy()
        X = batch_data[:, :-1]
        y = batch_data[:, -1]
        return X, y

def build_and_compile_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(1024, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.Dense(512, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.Dense(256, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.Dense(128, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.Dense(64, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.Dense(32, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.Dense(16, kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.PReLU(alpha_initializer=keras.initializers.Constant(value=0.25)))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    n_worker = 1
    use_multiprocessing = False
    max_queue_size = 100
    mp.set_start_method("spawn", force=True)

    data_train_generator = CustomDataGenerator(
        list_src_data=[r"shared_big_data_1/data_train_1",
                       r"shared_big_data_2/data_train_2"], 
        dataset_type="train",
        n_worker=n_worker,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size)
    data_validation_generator = CustomDataGenerator(
        list_src_data=[r"shared_big_data_2/data_validation"], 
        dataset_type="validation",
        n_worker=n_worker,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size)
    
    input_shape = data_train_generator[0][0].shape[1]
    model = build_and_compile_model(input_shape)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=r"TLMN_see_opponent_cards_3_players.keras",
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)

    history = model.fit(data_train_generator, 
                        validation_data=data_validation_generator,
                        callbacks=[model_checkpoint_callback], 
                        epochs=50)