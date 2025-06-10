import polars as pl
import os
import gc
import math
import keras
from keras import models, layers

class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self,
                src_data,
                batch_size, 
                dataset_type,
                shuffle=True,
                **kwargs):
        super().__init__(**kwargs)
        self.src_data = src_data
        self.batch_size = batch_size
        assert dataset_type in ["train", "validation", "test"]
        self.dataset_type = dataset_type
        self.shuffle = shuffle
        self.dataset = self.read_dataset()
    
    def read_dataset(self):
        dataset = pl.read_parquet(os.path.join(self.src_data, 
                                               f"data_{self.dataset_type}.parquet"))
        print(f"The shape of the dataset is ({dataset.shape[0]:,}; {dataset.shape[1]})")
        return dataset.sample(fraction=1.0, shuffle=True)

    def on_epoch_end(self):
        # print("Shuffling dataset on the end of epoch")
        # start_time_shuffle = time.time()
        if self.shuffle:
            self.dataset = self.dataset.sample(fraction=1.0, shuffle=True)
            gc.collect()
        # print(f"Shuffling took {time.time() - start_time_shuffle} seconds")

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def normalize_dataset(self, dataset: pl.DataFrame):
        bool_cols = [col for col in dataset.columns if dataset[col].dtype == pl.Boolean]
        dataset = dataset.with_columns(
            [
                pl.col(bool_cols).cast(pl.Int8),  # Convert boolean columns to Int8
                (pl.col("player1_action_max_card") + 1) / 52,  # Normalize player1_action_max_card
                (pl.col("action_max_card") + 1) / 52,  # Normalize action_max_card
                (pl.col("len_player1_card") - 1) / 12,  # Normalize len_player1_card
            ]
        )
        return dataset

    def __getitem__(self, index):
        # batch_data = sparse.load_npz(os.path.join(self.src_data, 
        #                     f"data_{self.dataset_type}_batch_{index}.npz")).toarray()
        batch_data = self.dataset[(index * self.batch_size): ((index + 1) * self.batch_size)]
        batch_data = self.normalize_dataset(batch_data)
        batch_data = batch_data.to_numpy()
        X = batch_data[:, :-1]
        y = batch_data[:, -1]
        return X, y


src_data = r"data"
batch_size = 64
data_train_generator = CustomDataGenerator(
    src_data=src_data,
    batch_size=batch_size,
    dataset_type="train")
data_validation_generator = CustomDataGenerator(
    src_data=src_data,
    batch_size=batch_size,
    dataset_type="validation")

model = models.Sequential()
model.add(layers.Input(shape=(data_train_generator[0][0].shape[1],)))
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
# model.add(layers.Dense(1, activation="linear"))
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=r"tmp_test.keras",
    monitor='val_mae',
    mode='min',
    save_best_only=True)

history = model.fit(data_train_generator, 
                    validation_data=data_validation_generator,
                    callbacks=[model_checkpoint_callback], epochs=50)