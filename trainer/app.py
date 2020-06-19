from __future__ import absolute_import
from flask import Flask, request
import json
import os
import requests
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
import sys

if sys.platform.startswith('win'):
    import eventlet
    eventlet.monkey_patch()


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://127.0.0.1:6381/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://127.0.0.1:6381/0'
celery_worker = Celery('worker', broker=app.config['CELERY_BROKER_URL'])

# TRAINER CLASS #

class Trainer(Thread):

    def __init__(self, trainable_queue, unlabelled_queue):
        Thread.__init__(self)
        self.train_queue = train_queue
        self.unlabelled_queue = unlabelled_queue
        self.started = False
    def init(self, input_shape, num_classes):
        
        self.num_classes = num_classes
        self.input_shape = input_shape 
        self.model = self.setup_model()
        self.started = True
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
        return np.sum(Entropy_vect(pred), dtype=np.float64)

    def sort_func(self, list_of_score_dicts):
        '''E/E strat, currently only decreasingly sorting by score'''
        return sorted(list_of_score_dicts, key = lambda i: (i["score"]), reverse=True)

    def send_sorted_data(self, data):
        r = requests.post("http://127.0.0.1:3334/retrieve_query", data=json.dumps(data))
        with open("error.html", "w") as f:
            f.write(r.text)
        with open("dumped.json", "w") as f:
            json.dump(data, f)

    def MakeQuery(self, unlabelled_set, 
                uncertainty_measure=SumEntropy, EEstrat=sort_func):    
        '''unlabelled_set : (image, filename) !
           uncertainty_measure : the higher the more uncertain
           return dict = {"filename", "score"} decreasingly sorted by score'''
        dict_keys = ["filename", "score"]
        list_of_score_dicts = []
        #TODO: Batching !
        for unlabelled_image, filename in unlabelled_set.as_numpy_iterator():
            unlabelled_image = np.expand_dims(unlabelled_image, axis=0)
            score_dict = dict.fromkeys(dict_keys)
            pred = self.model.predict(unlabelled_image)
            score_dict["score"] = uncertainty_measure(self, pred[0])
            score_dict["filename"] = filename.decode("utf-8") 
            list_of_score_dicts.append(score_dict)
        return EEstrat(self, list_of_score_dicts)

    def run(self):
        while not stopTrainer.is_set():
            print("Waiting for the feeder to feed us :'( ")
            train_set = self.train_queue.get()
            print("We got fed !! Resuming training now")
            self.model.fit(train_set, epochs=5)
            print("Retrieving the unlabelled set")
            unlabelled_set = self.unlabelled_queue.get()
            print("Making predictions ....")
            sorted_unlabelled_set = self.MakeQuery(unlabelled_set)  
            print("Sending query")   
            self.send_sorted_data(sorted_unlabelled_set) 
            
            
# UTILS #

def decode_img(file_path, model_input_shape=(224, 224)):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3) #TODO: Support for grayscale image
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, model_input_shape)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)


def training_set_creation(data, model_input_shape, num_classes):
    print(data)
    dataset = tf.data.Dataset.from_tensor_slices(tuple(data))     
    def pre_pro_training(file_path, label): 
        img = decode_img(file_path, model_input_shape=model_input_shape)
        label = tf.one_hot(label, depth=num_classes)
        return (img, label)
    return dataset.map(pre_pro_training)

def unlabelled_set_creation(data, model_input_shape):
    def pre_pro_unlabelled(file_path):
        img = decode_img(file_path, model_input_shape=model_input_shape)
        return (img, file_path)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset.map(pre_pro_unlabelled)




# CELERY TASKS # 

@celery_worker.task(name="stopper")
def stop_training():
    stopTrainer.set()

@celery_worker.task(name="feed_training")
def feed_training_data(data, labels_list):
    num_classes = len(labels_list)
    model_input_shape = (224, 224)
    if not trainer.started:
        trainer.init(input_shape=model_input_shape+(3,), num_classes=num_classes)
        trainer.start()

    train_set = training_set_creation(data, num_classes=num_classes, model_input_shape=model_input_shape)
    train_set = train_set.batch(4) #TODO: Batch size variable 
    train_queue.put(train_set)
    return "feed_training done"

@celery_worker.task(name="feed_query")
def feed_query_data(data):
    unlabelled_set = unlabelled_set_creation(data, model_input_shape=(224, 224)) 
    unlabelled_set.batch(4) #TODO: Variable batch size
    unlabelled_queue.put(unlabelled_set)
    return "feed_query done"

# @app.route("/")

@app.route('/train', methods=['POST'])
def retrieve_data():
    """Retrieve the images to feed to the trainer  Class

        reqs = {
            "labelled_data": [impaths, labels]
            "labels_list": self explanatory
            "init": boolean
            "unlabelled": [impaths] 
            }
    """
    data = json.loads(request.data)

    if not os.path.isdir("./assets/annotations"):
        os.mkdir("./assets/annotations")
    try:
        train_nb = len(os.listdir("./assets/annotations/"))
    except:
        train_nb = 0
    path_annot = os.path.join(f"./assets/annotations/annots_{str(train_nb)}.json")
    with open(path_annot, 'w') as f:
        json.dump(data, f)

    feed_training_data.delay(data["labelled_data"], data["labels_list"])
    feed_query_data.delay(data["unlabelled"])
    return "hello gary"


train_queue = queue.Queue()
unlabelled_queue = queue.Queue()
trainer = Trainer(train_queue, unlabelled_queue)
stopTrainer = Event()

if __name__ == '__main__':
    app.run(host="localhost", port=3333, debug=True)
    