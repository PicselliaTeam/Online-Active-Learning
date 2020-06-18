from __future__ import absolute_import
from flask import Flask 
import json
import celery 
import os
import requests
from flask import request
from celery import Celery
import sys
import PIL
import numpy as np
from redis import Redis 
from rq import Queue
from threading import Thread, Event
import queue
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from utils import get_diff

#TODO: if sys = windows then do this else no, see with docker blablaidk and check imports
import eventlet
eventlet.monkey_patch()


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://127.0.0.1:6381/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://127.0.0.1:6381/0'
celery = Celery('app', broker=app.config['CELERY_BROKER_URL'])
# celery.conf.update(app.config) #TODO: Check with thibault on ubuntu the use of this

# TRAINER CLASS #

class Trainer(Thread):

    def __init__(self, trainable_queue, unlabelled_queue, sorted_unlabelled_queue):
        Thread.__init__(self)
        self.train_queue = train_queue
        self.unlabelled_queue = unlabelled_queue
        self.sorted_unlabelled_queue = sorted_unlabelled_queue

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

    def SumEntropy(self, pred):
        def Entropy(prob):
            return -prob*np.log(prob)
        Entropy_vect = np.vectorize(Entropy)
        return np.sum(Entropy_vect(pred))

    def sort_func(self, list_of_score_dicts):
        '''E/E strat, currently only decreasingly sorting by score'''
        return sorted(list_of_score_dicts, key = lambda i: (i["score"]), reverse=True)

    def MakeQuery(self, unlabelled_set, 
                uncertainty_measure=SumEntropy, EEstrat=sort_func):    
        '''unlabelled_set : (image, filename) !
           uncertainty_measure : the higher the more uncertain
           return dict = {"filename", "score"} decreasingly sorted by score'''
        dict_keys = ["img_name", "score"]
        list_of_score_dicts = []
        for unlabelled_image, filename in unlabelled_set:
            score_dict = dict.fromkeys(dict_keys)
            pred = self.model.predict(unlabelled_image)          
            score_dict["score"].append(uncertainty_measure(pred))
            score_dict["filename"].append(filename)
            list_of_score_dicts.append(score_dict)
        return EEstrat(list_of_score_dicts)

    def run(self):
        while not stopTrainer.is_set():
            print("Waiting for the feeder to feed us :'( ")
            train_set = self.train_queue.get()
            print("We got fed !! Resuming training now")
            self.model.fit(train_set, epochs=5)
            print("Retrieving the unlabelled set")
            unlabelled_set = self.unlabelled_queue.get()
            print("Starting predictions ....")
            sorted_unlabelled_set = self.MakeQuery(unlabelled_set)         
            self.sorted_unlabelled_queue.put(sorted_unlabelled_set)
            print("Sending query")



    
# UTILS #

def decode_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3) #TODO: Support for grayscale image
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, input_shape)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)


def pre_pro_training(file_path, label):           
    img = decode_img(file_path)
    label = tf.one_hot(label, depth=num_classes)
    return (img, label)

def pre_pro_query(file_path): 
    '''use this later for batching'''
    img = decode_img(file_path)
    return (img, file_path)

def feed_training_data(data, num_classes):
    def dataset_creation(data, model_input_shape, num_classes):
        dataset = tf.data.Dataset.from_tensor_slices(tuple(data))
        return dataset.map(pre_pro_training)

    train_set = dataset_creation(data, model_input_shape=(224,224), num_classes=num_classes)
    train_set = train_set.batch(4) #TODO: Batch size variable 
    train_queue.put(train_set)

def feed_query_data(data):
    def dataset_creation(data, model_input_shape):
        data = tf.data.Dataset.from_tensor_slices(data)
        return dataset.map(decode_img)
    unlabelled_set = dataset_creation(data, model_input_shape=(224, 224)) 
    unlabelled_set.batch(4) #TODO: Variable batch size
    unlabelled_queue.put(unlabelled_set)


# CELERY TASKS # 

@celery.task
def stop_training():
    stopTrainer.set()

@celery.task
def start_training(data, labels_list, init):
    num_classes = len(labels_list)
    if init:
        trainer.init(input_shape=(224,224,3), num_classes=num_classes)
    feed_training_data(data, num_classes)
    if init:
        trainer.start()
    #TODO: Why two if statements


@app.route('/train', methods=['POST'])
def retrieve_data():
    """Retrieve the images to feed to the trainer  Class

        reqs = {
            "labelled_data": [impaths, labels]
            "labels_list": self explanatory
            "init": boolean
            "unlabelled_set": 
            }
    """
    data = json.loads(request.data)

    if not os.path.isdir("../assets/annotations"):
        os.mkdir("../assets/annotations")
    try:
        train_nb = len(os.listdir("../assets/annotations/"))
    except:
        train_nb = 0
    path_annot = os.path.join(f"../assets/annotations/annots_{str(train_nb)}.json")
    with open(path_annot, 'w') as f:
        json.dump(data, f)
    

    start_training.delay(data["labelled_data"], data["labels_list"], data["init"])
    
    return 'Hello, World!'


train_queue = queue.Queue()
unlabelled_queue = queue.Queue()
sorted_unlabelled_queue = queue.Queue()
trainer = Trainer(train_queue, unlabelled_queue, sorted_unlabelled_queue)
stopTrainer = Event()
isTraining = Event()

if __name__ == '__main__':
    app.run(host="localhost", port=3333, debug=True)
    