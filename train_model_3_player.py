import polars as pl
import os
import gc
import math
import keras
from keras import models, layers

class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self,
                src_data,
                dataset_type,
                **kwargs):
        super().__init__(**kwargs)
        self.src_data = src_data
        assert dataset_type in ["train", "validation", "test"]
        self.dataset_type = dataset_type

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(os.listdir(self.src_data))

    def normalize_dataset(self, dataset: pl.DataFrame):
        bool_cols = [col for col in dataset.columns if dataset[col].dtype == pl.Boolean]
        dataset = dataset.with_columns(
            [
                pl.col(bool_cols).cast(pl.Int8),  # Convert boolean columns to Int8
                (pl.col("player2_action_max_card") + 1) / 52,  # Normalize player2_action_max_card
                (pl.col("player3_action_max_card") + 1) / 52,  # Normalize player3_action_max_card
                (pl.col("action_max_card") + 1) / 52,  # Normalize action_max_card
            ]
        )
        return dataset

    def __getitem__(self, index):
        batch_data = pl.read_parquet(os.path.join(self.src_data, 
                        f"data_{self.dataset_type}_batch_{index}.parquet"))
        batch_data = self.normalize_dataset(batch_data)
        batch_data = batch_data.to_numpy()
        X = batch_data[:, :-1]
        y = batch_data[:, -1]
        return X, y

data_train_generator = CustomDataGenerator(
    src_data=r"folderdata/data_train", 
    dataset_type="train")
data_validation_generator = CustomDataGenerator(
    src_data=r"folderdata/data_validation", 
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

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=r"test_run_time.keras",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

history = model.fit(data_train_generator, 
                    validation_data=data_validation_generator,
                    callbacks=[model_checkpoint_callback], 
                    epochs=50)