import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import flask
import time
import os
import numpy as np
import json
import requests 
from flask import request
import random



## Server conf ##

server = flask.Flask('app')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('app', server=server, external_stylesheets=external_stylesheets)
image_directory = "assets/images/"


## Labeler class definition ##

class Labeler():
    def __init__(self, png_dir):
        self.png_dir = png_dir
        self.unlabelled = self.configure_dir(png_dir=png_dir)
        self.images_tosend = []
        self.ground_truths = []
        random.shuffle(self.unlabelled)
        # self.labels_list = self.configure_label()
        self.labels_list =  ["Morrowind", "Vignes", "xview"]
        self.update_iter()
        self.configure_labelmap()
        

    def configure_dir(self, png_dir):
        for file in os.listdir(png_dir):
            if not file.endswith((".png", ".jpg")):
                print("Your directory does not contain only images, please clean it :)")
                return
        print("Scanning {} ... OK".format(png_dir))
        return os.listdir(png_dir)

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
    
    def configure_labelmap(self):
        self.labelmap = {}
        for i, l in enumerate(self.labels_list):
            self.labelmap[l] = i

    def update_iter(self):
        [self.unlabelled.remove(p) for p in self.images_tosend if p in self.unlabelled]
        self.iter_images = np.nditer([self.unlabelled])
        print("iterator updated !")

    def send_data(self):
        '''reqs = {"labelled_data": [impaths, labels] 
                "labels_list": self explanatory
                "unlabelled": [impaths] }'''      
        path = "./labeler/assets/images"
        to_keep = self.images_tosend.pop()
        print("to keep", to_keep)
        print(len(self.unlabelled))
        [self.unlabelled.remove(p) for p in self.images_tosend if p in self.unlabelled]
        print(len(self.unlabelled))
        to_send_1 = [os.path.join(path, x) for x in self.images_tosend]
        to_send_2 = [os.path.join(path, x) for x in self.unlabelled]
        print(self.ground_truths, self.images_tosend)
        data = {"labelled_data": (to_send_1, self.ground_truths), "labels_list": self.labels_list, "unlabelled": to_send_2}
        with open("temp.json", "w") as f:
            json.dump(data, f)
        r = requests.post("http://localhost:3333/train", data=json.dumps(data))
        print("data sent!")
        self.ground_truths = []
        self.images_tosend = [to_keep]

labeler = Labeler(png_dir=image_directory)

## Layout, etc ##

static_image_route = "/static/"
center_style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}

app.layout = html.Div(
    [
        html.H1('Image to annotate', style=center_style),      
        html.Div([html.Button(name1, id='sbutton{}'.format(name1), n_clicks=0) for name1 in labeler.labels_list], style=center_style),
        html.Img(id='image', style=center_style), 
        html.Div(id="disp_sbutton", children="Labels Button")
    ]
)


## Flask routes ##

@server.route("/retrieve_query", methods=['POST'])
def retrieve_data():
    '''Retrieve the sorted unlabelled list of dicts of keys [filenames, scores]'''
    unlabelled_sorted_dict = json.loads(request.data)
    ## retrieve task here
    labeler.unlabelled = [os.path.split(x["filename"])[1] for x in unlabelled_sorted_dict]
    labeler.update_iter()
    print("data retrieved!", len(labeler.unlabelled))
    return "Hey"


@server.route(f'{static_image_route}<image_name>')
def serve_image(image_name):
    ## 
    if image_name not in labeler.unlabelled:
        raise Exception(f'"{image_name}" is excluded from the allowed static files')
    return flask.send_from_directory(image_directory, image_name)

@app.server.route("/stop_annotating")
def serve_stop_image():
    return flask.send_file("picsell_logo.png")

@app.callback([Output('image', 'src'),
            Output('disp_sbutton', 'children')],
     [Input('sbutton{}'.format(name1), 'n_clicks') for name1 in labeler.labels_list])
def update(*n_clicks):
    try:
        image = str(next(labeler.iter_images))
        labeler.images_tosend.append(image)
    except StopIteration:
        return ("/stop_annotating", html.Div('Fini'))

    msg = "No button was clicked"
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    ## append selected label task here
    for label in labeler.labels_list:
        if label in changed_id:
            selected_label = label
            labeler.ground_truths.append(labeler.labelmap[selected_label])
            msg = f'Label selected was {selected_label}'
    
    ## calculate len task here
    if BATCH_SIZE>len(labeler.ground_truths):
        print(labeler.ground_truths, labeler.images_tosend)
        return (static_image_route + image, html.Div(msg))

    else:
        labeler.send_data()
        return (static_image_route + image, html.Div(msg))
    
BATCH_SIZE = 4

if __name__ == '__main__':
    # app.run_server(debug=True, use_reloader=False) #to not launch twice everything
    app.run_server(debug=True, port=3334)
