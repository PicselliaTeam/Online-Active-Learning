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
import config

app = Flask(__name__)

if config.FORCE_ON_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# TRAINER CLASS #
class Trainer(Thread):

    def __init__(self, trainable_queue, unlabelled_queue, test_queue, daemon=True):
        Thread.__init__(self, daemon=daemon)

        self.train_queue = train_queue
        self.unlabelled_queue = unlabelled_queue
        self.test_queue = test_queue
        self.started = False
        self.query_countdown = 0
        self.eval_countdown = 0

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
        requests.post(config.LABELER_IP+"/retrieve_query", data=json.dumps(data))

    def make_query(self, data, EEstrat=config.ee_strat):
        dict_keys = ["filename", "score"]
        list_of_score_dicts = []
        dataset = data[0]
        filenames = data[1]
        preds = self.model.predict(dataset)
        for i,p in enumerate(preds):
            score_dict = dict.fromkeys(dict_keys)
            score_dict["score"] = p 
            score_dict["filename"] = filenames[i]
            list_of_score_dicts.append(score_dict)
        return EEstrat(list_of_score_dicts)

    def update_train_set(self, previous_train_set=None):
        if self.train_queue.qsize()>0 or previous_train_set==None:
            temp = []
            k = 0
            if self.train_queue.qsize()==0:
                print("Waiting for new training data")
            while self.train_queue.qsize() > 0 or k==0:
                k+=1
                self.query_countdown+=1
                self.eval_countdown+=1
                to_append = self.train_queue.get()
                temp.append(to_append)

            if not "stop" in temp:
                train_set = temp[0]
                if previous_train_set:
                    train_set.concatenate(previous_train_set)
                if len(temp)>1:
                    for k in range(len(temp)-1):
                        train_set.concatenate(temp[k+1])
                print(f"We got fed new training data! Number of requests before evaluation : {config.EVAL_EVERY-self.eval_countdown} and before query : {config.QUERY_EVERY-self.query_countdown}")

                return train_set
            if "stop" in temp:
                return "stop"
        else:
            return previous_train_set

    def update_unlabelled_data(self):
        k = 0
        while self.unlabelled_queue.qsize() > 0 or k==0:
            k+=1
            unlabelled_data = self.unlabelled_queue.get()
        return unlabelled_data

    def run(self):
        print("Waiting for the test set")
        test_set = self.test_queue.get()
        print("Test set acquired")
        while not stopTrainer.is_set():
            if self.first_iter:               
                train_set = self.update_train_set()
                if train_set == "stop":
                    print("Stopping before the first training data batch was received")
                    break
                self.first_iter = False
            else:
                if config.TRAIN_CONTINUOUSLY:
                    new_data = self.update_train_set(train_set)
                    if new_data == "stop":
                        break
                    train_set = new_data
                else:
                    to_concat = self.update_train_set()
                    if to_concat == "stop":
                        break
                    train_set.concatenate(to_concat)

                
            self.model.fit(train_set, epochs=config.NUM_EPOCHS_PER_LOOP, verbose=config.TRAINING_VERBOSITY)
            if self.unlabelled_queue.qsize() > 0:
                unlabelled_data = self.update_unlabelled_data()
                if  self.eval_countdown >= config.EVAL_EVERY:
                    print("Model evaluation")
                    self.eval_countdown = 0
                    evaluation = self.model.evaluate(test_set)
                    print("Evaluation result:")
                    for e,n in zip(evaluation, self.model.metrics_names):
                        print(f"{n} is {e}")
                        thresh = config.EARLY_STOPPING_METRICS_THRESHOLDS.get(n)
                        if thresh:
                            if thresh[1]=="upper_bound":
                                trigger = e>=thresh[0]
                            elif thresh[1]=="lower_bound":
                                trigger = e<=thresh[0]
                            if trigger:
                                print(f"Treshold ({thresh[0]}) reached for {n}")
                                stopTrainer.set()
                                requests.post(config.LABELER_IP+"/early_stopping", data={})

                    if not stopTrainer.is_set() and self.query_countdown >= config.QUERY_EVERY:
                        self.query_countdown = 0
                        print("Starting predictions")
                        sorted_unlabelled_data = self.make_query(unlabelled_data)  
                        print("Sending query")
                        self.send_sorted_data(sorted_unlabelled_data)
        print("Running last evaluation")
        evaluation = self.model.evaluate(test_set)
        print(evaluation)
        print("Stopping")
        self.model.save(config.SAVED_MODEL_PATH)
        print("Model saved, you can safely shut down the server")

# UTILS #
def decode_img(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3) #TODO: Support for grayscale image
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, config.INPUT_SHAPE)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img)

def save_test_data(data):
    if not os.path.isdir(config.ANNOTATIONS_SAVE_PATH):
        os.mkdir(config.ANNOTATIONS_SAVE_PATH)
    path = os.path.join(config.ANNOTATIONS_SAVE_PATH, "annotations.json")
    if os.path.isfile(path):
        return
    with open(path, "w") as f:
        json.dump(data, f)

 
def save_training_annotations(data):
    if not os.path.isdir(config.ANNOTATIONS_SAVE_PATH):
        os.mkdir(config.ANNOTATIONS_SAVE_PATH)
    path = os.path.join(config.ANNOTATIONS_SAVE_PATH, "annotations.json")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            data_json = json.load(f)
        if "labelled_data" in data_json:
            data_json["labelled_data"][0].extend(data["labelled_data"][0])
            data_json["labelled_data"][1].extend(data["labelled_data"][1])
        else:
            data_json["labelled_data"] = data["labelled_data"]
        data_json["unlabelled"] = data["unlabelled"]
    else:
        data_json = data
    with open(path, "w") as f:
        json.dump(data_json, f)


def dataset_set_creation(data, num_classes):
    dataset = tf.data.Dataset.from_tensor_slices(tuple(data))     
    def pre_pro_training(file_path, label): 
        img = decode_img(file_path)
        label = tf.one_hot(label, depth=num_classes)
        return (img, label)
    return dataset.map(pre_pro_training).shuffle(config.SHUFFLE_BUFFER_SIZE).batch(config.BATCH_SIZE)

def unlabelled_set_creation(data):
    def pre_pro_unlabelled(file_path):
        img = decode_img(file_path)
        return img
    dataset = tf.data.Dataset.from_tensor_slices(data)
    return [dataset.map(pre_pro_unlabelled).batch(config.BATCH_SIZE), data]

def feed_test_data(data, labels_list):
    num_classes = len(labels_list)
    test_set = dataset_set_creation(data, num_classes=num_classes)
    test_queue.put(test_set)


def feed_training_data(data, labels_list):
    num_classes = len(labels_list)
    train_set = dataset_set_creation(data, num_classes=num_classes)
    train_queue.put(train_set)

def feed_query_data(data):
    unlabelled_data = unlabelled_set_creation(data) 
    unlabelled_queue.put(unlabelled_data)

## Server routes ##


@app.route("/stop_training", methods=["POST"])
def stop_training():
    data = json.loads(request.data)
    if not data == {}:
        save_training_annotations(data)
    if trainer.started:
        stopTrainer.set()
        if trainer.first_iter == True:
            stop_msg = "stop"
            #TODO: Add dummy test queue filler in case stopping before having send test data
            train_queue.put(stop_msg)
        trainer.join()
    return "Training Stopped"

@app.route("/init_training", methods=["POST"]) ## No need for async since need to wait for it
def send_init_sig():
    '''Init the worker thread'''
    data = json.loads(request.data)
    labels_list = data["labels_list"]
    num_classes = len(labels_list)
    if not trainer.started:
        model = config.model_fn(num_classes)
        trainer.init(model)
        trainer.start()
    return "Trainer initialized"

@app.route('/train', methods=['POST'])
def retrieve_data():
    """Retrieve the images to feed to the trainer  Class

        reqs = {
            "labelled_data": [impaths, labels]
            "labels_list": self explanatory
            "unlabelled": [impaths] 
            }
    """
    data = json.loads(request.data)
    save_training_annotations(data)
    feed_training_data(data["labelled_data"], data["labels_list"])
    if len(data["unlabelled"]) > 0:
        feed_query_data(data["unlabelled"])
    return ""

@app.route("/test_data", methods=["POST"])
def test_data():
    '''Retrieve the test data and send them to the trainer thread
        data = {"test_data":, "labels_list":}
    '''
    data = json.loads(request.data)
    save_test_data(data)
    feed_test_data(data["test_data"], data["labels_list"])
    return ""

## Queue, events and thread def ##

train_queue = queue.Queue()
test_queue = queue.Queue()
unlabelled_queue = queue.Queue()
trainer = Trainer(train_queue, unlabelled_queue, test_queue)
stopTrainer = Event()

if __name__ == '__main__':
    app.run(host="localhost", port=3333)
    