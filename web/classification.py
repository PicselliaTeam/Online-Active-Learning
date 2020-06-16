import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from threading import Thread, Event
import queue
import time

class Trainer(Thread):

    def __init__(self,  q):
        Thread.__init__(self)
        self.q = q

    def init(self, input_shape, num_classes):
        
        self.num_classes = num_classes
        self.input_shape = input_shape 
        self.model = self.setup_model()
    def setup_model(self):
        baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=self.input_shape,
            input_tensor=layers.Input(shape=self.input_shape))



        headModel = baseModel.output
        headModel = layers.AveragePooling2D(pool_size=(3, 3))(headModel)
        headModel = layers.Flatten(name="flatten")(headModel)
        headModel = layers.Dense(128, activation="relu")(headModel)
        headModel = layers.Dropout(0.5)(headModel)
        headModel = layers.Dense(self.num_classes, activation="softmax")(headModel)

        baseModel.trainable = False


        model = keras.Model(inputs=baseModel.input, outputs=headModel)
        model.compile(loss='binary_crossentropy',
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
        return model

    def run(self):
        while not stopTrainer.is_set():
            print("Waiting for feeder to feed us :'( ")
            train_set = self.q.get()
            print("We got fed !!\n")
            for element in train_set:
                print(element)
            self.model.fit(train_set, epochs=5)