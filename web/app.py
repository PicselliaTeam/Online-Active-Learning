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
from classification import Trainer
from utils import get_diff


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6380/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6380/0'
celery = Celery('app', broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
q = queue.Queue()

# CELERY TASKS # 

@celery.task
def stop_training():

    stopTrainer.set()


@celery.task
def start_training(id, init):
    cnt = 0

    path = "../assets/annotations/"+str(id)

    no = len(os.listdir(path))-1
    
    if no != 0:
        old_path = os.path.join(path, str(no-1))+'.json'
        new_path = os.path.join(path, str(no))+'.json'
        dict_annotations = get_diff(old_path, new_path)

    else:
        with open(os.path.join(path, str(no))+'.json', 'r') as f:
            dict_annotations = json.load(f)

    png_dir = os.path.join("../assets/pictures", id)  
    print("Downloading PNG images to your machine ...")

    dl = 0
    total_length = len(dict_annotations["images"])
    for info in dict_annotations["images"]:

        pic_name = os.path.join(png_dir, info['external_picture_url'].split('/')[-1])
        if not os.path.isdir(png_dir):
            os.makedirs(png_dir)
        if not os.path.isfile(pic_name):
            try:
                response = requests.get(info["signed_url"], stream=True)
                with open(pic_name, 'wb') as handler:
                    for data in response.iter_content(chunk_size=1024):
                        handler.write(data)
                cnt += 1
            except:
                print("Image %s can't be downloaded" % pic_name)
                pass

        dl += 1
        done = int(50 * dl / total_length)
        sys.stdout.flush()
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
    
    if init:
        trainer.init(input_shape=(224,224,3), num_classes=len(dict_annotations["categories"]))
        
    
    feed_data(path=os.path.join(path, str(no))+'.json')

    if init:
        trainer.start()


# UTILS #
def feed_data(path):

    with open(path, 'r') as f:
        dic = json.load(f)

    data = ([], [])
    id = path.split('/')[-2]
    num_classes = len(dic["categories"])
    labelmap = {}
    for i,l in enumerate(dic["categories"]):
        labelmap[l["name"]]=i

    for im, ann  in zip(dic["images"],dic["annotations"]):
        imgpath = os.path.split(im["external_picture_url"])[-1]
    
        realpath = os.path.join("../assets/pictures/",str(id), imgpath)
        img = PIL.Image.open(realpath).resize((224,224))

       
        data[0].append(np.array(img).astype(float))
        data[1].append(ann["annotations"][0]["label"])

    
    def dataset_creation(data, input_shape, num_classes, labelmap):
        images, labels = data

        tmp_labels = []

        for e in labels:
            tmp_labels.append(labelmap[e])

        labels = tmp_labels    
        to_feed=[]

        for im, l in zip(images,labels):
            to_feed.append((im,l))
        
        
        dataset = tf.data.Dataset.from_tensor_slices((images,labels))

        def map_images(image, label):
            label = tf.one_hot(label, depth=num_classes)
            image = tf.image.resize(image, (224,224))
            image = tf.cast(image, tf.float32)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            return (image, label)

        
        return dataset.map(map_images)

    train_set = dataset_creation(data, input_shape=(224,224,3), num_classes=num_classes, labelmap=labelmap)
    

    train_set = train_set.batch(4) 
    
    q.put(train_set)




@app.route('/train', methods=['POST'])
def retrieve_data():
    """Retrieve the images to feed to the trainer  Class

        reqs = {
            "annotations": annotations
            "project_id": id
        }
    """
    data = json.loads(request.data)

    annotations = data["annotations"]

    if not os.path.isdir("../assets/annotations"):
        os.mkdir("../assets/annotations")
    
    id = data["project_id"]

    try:
        no = len(os.listdir(os.path.join("../assets/annotations/", str(id))))
    except:
        os.makedirs(os.path.join("../assets/annotations/", str(id)))
        no = 0
    with open(os.path.join("../assets/annotations/", str(id),'{}.json'.format(str(no))), 'w') as f:
        json.dump(annotations, f)
    
    print(id)
    start_training.delay(id, data["init"])
    
    return 'Hello, World!'


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
            self.model.fit(train_set, epochs=5)


trainer = Trainer(q)
stopTrainer = Event()
isTraining = Event()


if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=3333, debug=True)
    