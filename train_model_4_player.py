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
                src_data,
                dataset_type,
                n_worker,
                use_multiprocessing,
                max_queue_size,
                **kwargs):
        super().__init__(**kwargs)
        self.src_data = src_data
        assert dataset_type in ["train", "validation", "test"]
        self.dataset_type = dataset_type
        self.workers = n_worker
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size
        self.bool_cols = self.__get_boolean_columns()

    def __get_boolean_columns(self):
        sample_data_dir = self.file_index[0]
        schema = pl.scan_parquet(sample_data_dir).collect_schema()
        bool_cols = [col for col, dtype in schema.items() if dtype == pl.Boolean]
        assert len(bool_cols) == len(schema) - 5
        return bool_cols

    def __len__(self):
        return len(os.listdir(self.src_data))

    def normalize_dataset(self, dataset: pl.DataFrame):
        dataset = dataset.with_columns(
            [
                pl.col(self.bool_cols).cast(pl.Float64),  # Convert boolean columns to Float64
                ((pl.col("player2_action_max_card") + 1) / 52).cast(pl.Float64),  # Normalize player2_action_max_card
                ((pl.col("player3_action_max_card") + 1) / 52).cast(pl.Float64),  # Normalize player3_action_max_card
                ((pl.col("player4_action_max_card") + 1) / 52).cast(pl.Float64),  # Normalize player4_action_max_card
                ((pl.col("action_max_card") + 1) / 52).cast(pl.Float64),  # Normalize action_max_card
            ]
        )
        return dataset

    def __getitem__(self, index):
        batch_data = pl.read_parquet(os.path.join(self.src_data, 
                                    f"data_{self.dataset_type}_batch_{index}.parquet"))
        batch_data = self.normalize_dataset(batch_data)
        batch_data = batch_data.to_numpy()
        assert all(x >= 0 and x <= 1 for x in batch_data.flatten())
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
    max_queue_size = 10

    data_train_generator = CustomDataGenerator(
        src_data="", 
        dataset_type="train",
        n_worker=n_worker,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size)
    data_validation_generator = CustomDataGenerator(
        src_data="", 
        dataset_type="validation",
        n_worker=n_worker,
        use_multiprocessing=use_multiprocessing,
        max_queue_size=max_queue_size)
    
    input_shape = data_train_generator[0][0].shape[1]
    model = build_and_compile_model(input_shape)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=r"TLMN_see_opponent_cards_4_players.keras",
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1)

    history = model.fit(data_train_generator, 
                        validation_data=data_validation_generator,
                        callbacks=[model_checkpoint_callback], 
                        epochs=50)