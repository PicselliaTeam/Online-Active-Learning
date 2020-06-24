from __future__ import absolute_import
from flask import Flask, request
import json
import os
import requests
import numpy as np
from threading import Thread, Event
import queue
import tensorflow as tf
import os
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.applications import MobileNetV2
import config

app = Flask(__name__)


# TRAINER and FEEDER CLASSES #
class Trainer(Thread):

    def __init__(self, trainable_queue, unlabelled_queue, test_queue, daemon=True):
        Thread.__init__(self, daemon=daemon)
        self.train_queue = train_queue
        self.unlabelled_queue = unlabelled_queue
        self.test_queue = test_queue
        self.started = False
        self.eval_and_query_countdown = 0

    def init(self, model):
        self.model = model
        self.started = True
        self.first_iter = True

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

    def update_train_set(self, previous_train_set=None):
        if self.train_queue.qsize()>0 or previous_train_set==None:
            l = []
            k = 0
            while self.train_queue.qsize() > 0 or k==0:
                k+=1
                self.eval_and_query_countdown+=1
                l.append(self.train_queue.get())

            train_set = l[0]
            if len(l)>1:
                for k in range(len(l)-1):
                    train_set.concatenate(l[k+1])
            print("We got fed !")
            return train_set
        else:
            return previous_train_set

    def update_unlabelled_set(self):
        k = 0
        while self.unlabelled_queue.qsize() > 0 or k==0:
            k+=1
            unlabelled_set = self.unlabelled_queue.get()
        return unlabelled_set

    def run(self):
        print("Waiting for the test set")
        test_set = self.test_queue.get()
        print("Test set acquired")
        while not stopTrainer.is_set():
            if self.first_iter:
                self.first_iter = False
                train_set = self.update_train_set()
            else:
                train_set.concatenate(self.update_train_set(train_set))
            self.model.fit(train_set, epochs=config.NUM_EPOCHS_PER_LOOP, verbose=config.TRAINING_VERBOSITY)
            if self.unlabelled_queue.qsize() > 0:
                unlabelled_set = self.update_unlabelled_set()
                if self.eval_and_query_countdown >= config.EVAL_AND_QUERY_EVERY:
                    print("Model evaluation")
                    self.eval_and_query_countdown = 0
                    evaluation = self.model.evaluate(test_set)
                    print("Evaluation result:")
                    for e,n in zip(evaluation, self.model.metrics_names):
                        print(f"{n} is {e}")
                    print("Starting predictions")
                    sorted_unlabelled_set = self.MakeQuery(unlabelled_set)  
                    print("Sending query")   
                    self.send_sorted_data(sorted_unlabelled_set) 
        print("Stopping")
        self.model.save("saved_model")
        print("Model saved, you can safely shut down the server")

# UTILS #
def decode_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3) #TODO: Support for grayscale image
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, config.INPUT_SHAPE)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)


def dataset_set_creation(data, num_classes):
    dataset = tf.data.Dataset.from_tensor_slices(tuple(data))     
    def pre_pro_training(file_path, label): 
        img = decode_img(file_path)
        label = tf.one_hot(label, depth=num_classes)
        return (img, label)
    return dataset.map(pre_pro_training)

def unlabelled_set_creation(data):
    def pre_pro_unlabelled(file_path):
        img = decode_img(file_path)
        return (img, file_path)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return dataset.map(pre_pro_unlabelled) 

def feed_test_data(data, labels_list):
    num_classes = len(labels_list)
    test_set = dataset_set_creation(data, num_classes=num_classes)
    test_set = test_set.batch(config.BATCH_SIZE)
    test_queue.put(test_set)


def feed_training_data(data, labels_list):
    num_classes = len(labels_list)
    train_set = dataset_set_creation(data, num_classes=num_classes)
    train_set = train_set.batch(config.BATCH_SIZE)
    train_queue.put(train_set)

def feed_query_data(data):
    unlabelled_set = unlabelled_set_creation(data) 
    unlabelled_set.batch(config.BATCH_SIZE)
    unlabelled_queue.put(unlabelled_set)

## Server routes ##


@app.route("/stop_training", methods=["POST"])
def stop_training():
    stopTrainer.set()
    trainer.join()
    return "Training Stopped"

@app.route("/init_training", methods=["POST"]) ## No need for async since need to wait for it
def send_init_sig():
    '''Init the worker thread'''
    data = json.loads(request.data)
    labels_list = data["labels_list"]
    num_classes = len(labels_list)
    if not trainer.started:
        model = config.setup_model(num_classes)
        trainer.init(model)
        trainer.start()
    return "Trainer initialized"

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
    if not os.path.isdir(config.ANNOTATIONS_PATH):
        os.mkdir(config.ANNOTATIONS_PATH)
    path = os.path.join(config.ANNOTATIONS_PATH, "annotations.json")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data_json = json.load(f)
        data_json["labelled_data"][0].extend(data["labelled_data"][0])
        data_json["labelled_data"][1].extend(data["labelled_data"][1])
        data_json["unlabelled"] = data["unlabelled"]
    else:
        data_json = data
    with open(path, "w") as f:
        json.dump(data_json, f)

    feed_training_data(data["labelled_data"], data["labels_list"])
    feed_query_data(data["unlabelled"])
    return ""

@app.route("/test_data", methods=["POST"])
def test_data():
    '''Retrieve the test data and send them to the trainer thread
        data = {"test_data":, "labels_list":}
    '''
    data = json.loads(request.data)
    feed_test_data(data["test_data"], data["labels_list"])
    return ""

## Queue, events and thread def ##

train_queue = queue.Queue()
test_queue = queue.Queue()
unlabelled_queue = queue.Queue()
trainer = Trainer(train_queue, unlabelled_queue, test_queue)
stopTrainer = Event()

if __name__ == '__main__':
    app.run(host="localhost", port=3333, debug=True, use_reloader=False)
    