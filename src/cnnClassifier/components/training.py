import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
import time


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        # Ensure functions run eagerly before loading
        import tensorflow as tf
        tf.config.run_functions_eagerly(True)
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        # Compile the model to ensure it's trainable
        # Use categorical loss because model outputs 2-class probabilities
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    def train_valid_generator(self):

        # Use tf.data API for compatibility with eager execution
        import tensorflow as tf
        self.train_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="training",
            seed=123,
            image_size=tuple(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            label_mode='int'  # produce integer labels, we'll one-hot them to match model output
        )

        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="validation",
            seed=123,
            image_size=tuple(self.config.params_image_size[:-1]),
            batch_size=self.config.params_batch_size,
            label_mode='int'
        )

        # Normalize images
        normalization_layer = tf.keras.layers.Rescaling(1./255)
        self.train_generator = self.train_generator.map(lambda x, y: (normalization_layer(x), y))
        self.valid_generator = self.valid_generator.map(lambda x, y: (normalization_layer(x), y))

        # Convert integer labels to one-hot vectors matching model output
        num_classes = int(self.model.output_shape[-1]) if self.model is not None else 2
        self.train_generator = self.train_generator.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)))
        self.valid_generator = self.valid_generator.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=num_classes)))

        # Optional augmentation
        if self.config.params_is_augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
            ])
            self.train_generator = self.train_generator.map(lambda x, y: (data_augmentation(x, training=True), y))

    @staticmethod
    def save_model(path: Path, model: 'tf.keras.Model'):
        model.save(path)


    def train(self, callback_list: list):
        # Use Dataset directly with Keras
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            callbacks=callback_list
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )