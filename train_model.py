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
        self.bool_cols = self.__get_boolean_columns()

    def __get_boolean_columns(self):
        schema = pl.scan_parquet(os.path.join(self.src_data, 
                                f"data_{self.dataset_type}.parquet")).collect_schema()
        bool_cols = [col for col, dtype in schema.items() if dtype == pl.Boolean]
        assert len(bool_cols) == len(schema) - 1
        return bool_cols

    def read_dataset(self):
        dataset = pl.read_parquet(os.path.join(self.src_data, 
                                               f"data_{self.dataset_type}.parquet"))
        print(f"The shape of the dataset {self.dataset_type} is ({dataset.shape[0]:,}; {dataset.shape[1]})")
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
        dataset = dataset.with_columns(
            [
                pl.col(self.bool_cols).cast(pl.Float64),  # Convert boolean columns to Float
            ]
        )
        return dataset

    def __getitem__(self, index):
        batch_data = self.dataset[(index * self.batch_size): ((index + 1) * self.batch_size)]
        batch_data = self.normalize_dataset(batch_data)
        batch_data = batch_data.to_numpy()
        X = batch_data[:, :-1]
        y = batch_data[:, -1]
        return X, y


src_data = r"D:\Project\TLMN\preprocess"
batch_size = 512
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
model.add(layers.Dense(1, activation="sigmoid"))
model.summary()

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=r"TLMN_2_players_honestly_new.keras",
    monitor='val_mae',
    mode='min',
    save_best_only=True,
    verbose=1)

history = model.fit(data_train_generator, 
                    validation_data=data_validation_generator,
                    callbacks=[model_checkpoint_callback], 
                    epochs=50)