import os 
import json
import numpy as np
import requests 
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt


def configure_dir(png_dir='../assets/images/'):
    for file in os.listdir(png_dir):
        if not file.endswith((".png", ".jpg")):
            print("Your directory does not contain only images, please clean it :)")
            return
    print("Scanning {} ... OK".format(png_dir))
    return png_dir


def configure_label():
    Continue = True
    label = []
    i=1
    while Continue:
        l = input("Label n°{} : ".format(i))
        if type(l) is str:
            label.append(l)
        a = input("Press enter to add a new label or type in something to stop the labelling ")
        if a != "":
            Continue=False
        i+=1
    return label

def get_10per_dataset(png_dir):
    listdir = os.listdir(png_dir)
    per10 = listdir[:int(0.1*len(listdir))]
    return [os.path.join(png_dir, e) for e in per10]



def labeler(to_label, labels):
    print(f"Your labels are: {labels}")
    diff = {}
    diff["categories"] = []
    diff["infos"]= []
    diff["images"]= []
    diff["annotations"]= []
    labelmap = {}
    for i, l in enumerate(labels_list):
        labelmap[l] = i
    ground_truths = []
    def submit(text):
        plt.close()
        print(text)
        return text
        
    for img in to_label: 
        im = plt.imread(img)
        plt.imshow(im)
        axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
        text_box = TextBox(axbox, 'Label', initial="")
        text_box.on_submit(lambda x: [ground_truths.append((labelmap[x])), plt.close()]) #TODO: Check class
        plt.show()
    data = {"labelled_data": (to_label, ground_truths), "labels_list": labels}
    return data

def send_data(data, init=True):
    data["init"] = init
    r = requests.post("http://localhost:3333/train", data=json.dumps(data))
    
# path = configure_dir()
# to_label = get_10per_dataset(path)
# labels_list = configure_label()
# data = labeler(to_label, labels_list)


# data["init"]=True
# to_save = data
# with open("temp.json", "w") as f:
#     json.dump(to_save, f)

with open("temp.json", "r") as f:
    data = json.load(f)


send_data(data, init=False)

