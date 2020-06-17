import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from threading import Thread, Event
import queue
import time
from PIL import Image
import string
train, test = keras.datasets.mnist.load_data() 

def dataset_creation(data, size, input_shape):
    images, labels = data
    images = images.reshape((size, 28, 28, 1)).astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    def map_prepro(image, label):
        image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, input_shape[:-1])
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return (image, label)
    return dataset.map(map_prepro)


splited_train = [(train[0][30000:], train[1][30000:]), (train[0][:30000], train[1][:30000])]


BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 500
INPUT_SHAPE = (96, 96, 3)
NUM_CLASSES = 10

train_sets = []
for train in splited_train:
    train_set = dataset_creation(train, size=30000, input_shape=INPUT_SHAPE)
    train_set = train_set.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    train_sets.append(train_set)
test_set = dataset_creation(test, size=10000, input_shape=INPUT_SHAPE)


baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE,
	input_tensor=layers.Input(shape=INPUT_SHAPE))


headModel = baseModel.output
headModel = layers.AveragePooling2D(pool_size=(3, 3))(headModel)
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(128, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(NUM_CLASSES, activation="softmax")(headModel)

baseModel.trainable = False


model = keras.Model(inputs=baseModel.input, outputs=headModel)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

train_set = train_sets[0]
history = model.fit(train_set,
                    epochs=1,
                    validation_data=test_set.batch(BATCH_SIZE))


filename_input = layers.Input(shape=(), dtype=tf.string)
model2 = keras.Model(inputs=[model.input, filename_input], 
                        outputs=[model.output, filename_input])

data, _ = test
data = data[:10, :, :]
df = []
for image in data:
    img = Image.fromarray(image)
    img = img.convert("RGB")
    img = img.resize((96, 96))
    df.append(np.array(img))
filenames = np.linspace(1, 10, 10, dtype=np.int64)

import random
b = [(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))) for k in range(10)]

print(b)
a = model2.predict([np.array(df), np.array(b)])
print(a)
# test_scores = model.evaluate(test_set, verbose=2)
# print('Test loss:', test_scores[0])
# print('Test accuracy:', test_scores[1])
# input("waiting next batch, please press enter to continue: ")


class Trainer(Thread):

    def __init__(self, model, test_set, q):
        Thread.__init__(self)
        self.model = model
        self.test_set = test_set
        self.q = q

    def run(self):
        while not stopTrainer.is_set():
            print("Waiting for feeder to feed us :'( ")
            self.q.get(train_set)
            print("We got fed !!\n")
            self.model.fit(train_set, epochs=5, validation_data=test_set)
            test_scores = self.model.evaluate(test_set, verbose=2)
            # print('Test loss:', test_scores[0])
            # print('Test accuracy:', test_scores[1])


class DataFeeder(Thread):

    def __init__(self, splited_train, q):
        Thread.__init__(self)
        self.splited_train = splited_train
        self.q = q 
    def run(self):
        for train in self.splited_train:
            train_set = dataset_creation(train, size=30000, input_shape=INPUT_SHAPE)
            train_set = train_set.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
            time.sleep(10)
            self.q.put(train_set)
        stopTrainer.set()

# q = queue.Queue()
# trainer = Trainer(model, test_set, q)
# feeder = DataFeeder(splited_train, q)

# stopTrainer = Event()
# isTraining = Event()
# print("starting multiple fits")
# trainer.start()
# feeder.start()
# trainer.join()
# feeder.join()

# print("Threads joined") 