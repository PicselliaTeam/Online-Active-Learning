import os 
import json
import numpy as np
import requests 
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
from flask import Flask, request
from celery import Celery

# app = Flask(__name__)
# app.config['CELERY_BROKER_URL'] = 'redis://127.0.0.1:6382/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://127.0.0.1:6382/0'
# celery_worker = Celery('worker', broker=app.config['CELERY_BROKER_URL'])

class Labeler():
    def __init__(self, png_dir='./assets/images/'):
        self.unlabelled =  self.configure_dir(png_dir=png_dir)
        self.labels_list = self.configure_label()
        self.to_label = []
        self.update_sets()
        self.annotate()

    def configure_dir(self, png_dir):
        for file in os.listdir(png_dir):
            if not file.endswith((".png", ".jpg")):
                print("Your directory does not contain only images, please clean it :)")
                return
        print("Scanning {} ... OK".format(png_dir))
        return [os.path.join(png_dir, p) for p in os.listdir(png_dir)]

    def configure_label(self):
        Continue = True
        labels_list = []
        i=1
        while Continue:
            l = input("Label n°{} : ".format(i))
            if type(l) is str:
                labels_list.append(l)
            a = input("Press enter to add a new label or type in something to stop the labelling ")
            if a != "":
                Continue=False
            i+=1
        return labels_list

    def annotate(self):
        '''used attributes: labels_list, to_label, '''
        print(f"Your labels are: {self.labels_list}")
        labelmap = {}
        for i, l in enumerate(self.labels_list):
            labelmap[l] = i
        ground_truths = []
        def submit(text):
            plt.close()
            print(text)
            return text

        for img in self.to_label: 
            while True:
                try:
                    im = plt.imread(img)
                    plt.imshow(im)
                    axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
                    text_box = TextBox(axbox, 'Label', initial="")                
                    text_box.on_submit(lambda x: [ground_truths.append((labelmap[x])), plt.close()]) #TODO: Check class
                    plt.show()
                    break
                except KeyError as k: 
                    print("Wrong label !")

        annotations = {"labelled_data": (self.to_label, ground_truths), "labels_list": self.labels_list}
        return annotations

    def update_sets(self):
        number_tolabel = int(0.1*len(self.to_label+self.unlabelled))
        self.to_label = self.unlabelled[:number_tolabel]
        self.unlabelled = [self.unlabelled.remove(p) for p in self.to_label]
           
    def send_data(self):
        '''reqs = {"labelled_data": [impaths, labels]
                "labels_list": self explanatory
                "unlabelled": [impaths] }'''
        
        r = requests.post("http://localhost:3333/train", data=json.dumps(data))






# @celery_worker.task(name="annotate")
# def annotate_task():
#     labeler.annotate()
     


# @app.route("/retrieve_query", methods=['POST'])
# def retrieve_data():
#     '''Retrieve the sorted unlabelled list of dicts of keys [filenames, scores]'''
#     unlabelled_sorted_dict = json.loads(request.data)
#     labeler.unlabelled = [x["img_name"] for x in unlabelled_sorted_dict]
#     update_sets(labeler.to_label, labeler.unlabelled)
#     return ""

labeler = Labeler()



# # path = configure_dir()
# # to_label, unlabelled = init_get_10per_dataset(path)
# # labels_list = configure_label()
# # data = annotate(to_label, labels_list)
# # data["unlabelled"] = unlabelled

#  #


# # data["init"]=True
# # to_save = data
# # with open("temp.json", "w") as f:
# #     json.dump(to_save, f)


# if __name__ == '__main__':
#     app.run(host="localhost", port=3334, debug=True)
#     with open("temp.json", "r") as f:
#         data = json.load(f)
#     send_data(data)

