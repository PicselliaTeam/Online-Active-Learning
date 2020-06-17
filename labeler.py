import os 
os.environ['DISPLAY'] = ':0'
import json
import numpy as np
import requests 
from matplotlib.widgets import TextBox
def configure_dir(png_dir='./assets/pictures/8a6ce063-7ff5-41af-b733-09cee71c54aa/'):
    
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
        a = input("Type enter to add a new label, type a leter for stop")
        if a != "":
            Continue=False
        i+=1
    return label

def get_10per_dataset(png_dir):
    listdir = os.listdir(png_dir)
    
    per10 = listdir[:int(0.1*len(listdir))]
    
    return [os.path.join(png_dir, e) for e in per10]

path =configure_dir()
labels = configure_label()


to_label = get_10per_dataset(path)

import matplotlib.pyplot as plt
import cv2

def labeler(to_label, labels):
    print(labels)
    diff = {}
    diff["categories"] = []
    diff["infos"]= []
    diff["images"]= []
    diff["annotations"]= []
    
    ground_truth = []
    def submit(text):
        plt.close()
        print(text)
        return text
        

    for img in to_label:
        
        im = plt.imread(img)
        plt.imshow(im)
        axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
        text_box = TextBox(axbox, 'Label', initial="")
        text_box.on_submit(lambda x: [ground_truth.append((img,x)), plt.close()])
        plt.show()

    
    
    return ground_truth
    

x = labeler(to_label, labels)

# def format_output(data):


print(x)