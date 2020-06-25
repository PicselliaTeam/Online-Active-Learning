import dash
from dash.dependencies import Input, Output, State, ALL
import dash_core_components as dcc
import dash_html_components as html
import flask
import os
import numpy as np
import json
import requests 
from flask import request
import random
from threading import Thread
import queue
import config

## Server conf ##
server = flask.Flask('app')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('app', server=server, external_stylesheets=external_stylesheets)

## Labeler class definition ##

class Labeler():
    def __init__(self, png_dir):
        self.png_dir = png_dir
        self.unlabelled = self.configure_dir(png_dir=png_dir)
        self.images_tosend = []
        self.ground_truths = []
        random.shuffle(self.unlabelled)
        # self.test_set = self.unlabelled[:int(config.TEST_SET_FRAC*len(self.unlabelled))] 
        self.test_set = self.unlabelled[:10] 
        self.test_set_gt = []
        self.test_set_iter = np.nditer([self.test_set])
        self.test_set_done = False
        [self.unlabelled.remove(p) for p in self.test_set if p in self.unlabelled]
        self.labels_list = ["a"]
        self.update_iter()
        self.labels_selected = False
        self.trainer_inited = False

    def configure_dir(self, png_dir):
        for file in os.listdir(png_dir):
            if not file.endswith((".png", ".jpg")):
                raise ValueError("Your PNG_DIR does not contain only .png or .jpg, please clean it")
                
        print("PNG dir OK")
        return os.listdir(png_dir)
    
    def configure_labelmap(self):
        self.labelmap = {}
        for i, l in enumerate(self.labels_list):
            self.labelmap[l] = i

    def update_iter(self):
        [self.unlabelled.remove(p) for p in self.images_tosend if p in self.unlabelled]
        self.iter_images = np.nditer([self.unlabelled])

    def prep_send_data(self):
        '''reqs = {"labelled_data": [impaths, labels] 
                "labels_list": self explanatory
                "unlabelled": [impaths] }'''      
        path = os.path.join("labeler", config.IMAGE_DIRECTORY)
        to_keep = self.images_tosend.pop()
        [self.unlabelled.remove(p) for p in self.images_tosend if p in self.unlabelled]
        print(f"Number of images to annotate remaining: {len(self.unlabelled)}")
        to_send_1 = [path+x for x in self.images_tosend]
        to_send_2 = [path+x for x in self.unlabelled]
        data = {"labelled_data": (to_send_1, self.ground_truths), "labels_list": self.labels_list, "unlabelled": to_send_2}
        self.ground_truths = []
        self.images_tosend = [to_keep]
        return data

class Sender(Thread):
    def __init__(self, q_send, daemon=True):
        Thread.__init__(self, daemon=daemon)
        self.q_send = q_send

    def run(self):
        while True:
            data = q_send.get()
            r = requests.post("http://localhost:3333/train", data=json.dumps(data))
            print("Data sent to trainer")

class SendTestSet(Thread):
    def __init__(self, test_queue, daemon=True):
        Thread.__init__(self, daemon=daemon)
        self.test_queue = test_queue
    def run(self):
        data = self.test_queue.get()
        r = requests.post("http://localhost:3333/test_data", data=json.dumps(data))

## Object and threads init ##
labeler = Labeler(png_dir=os.path.join("labeler", config.IMAGE_DIRECTORY))
q_send = queue.Queue()
test_queue = queue.Queue()
sender = Sender(q_send)
sender.start()
test_set_sender = SendTestSet(test_queue)
test_set_sender.start()

## Dash Layouts ##
static_image_route = "/static/"
center_style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
image_style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "object-fit": "scale-down", "height":"90vh"}


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')])


def annotation_layout():
    return html.Div([                        
                    html.Div([html.Button(name1, id={'role': 'label-button', 'index': name1}, n_clicks=0) for name1 in labeler.labels_list], style=center_style),
                    html.Div([html.Img(id='image', style=image_style)], style=center_style),
                    html.Div(html.Button("stop-signal", id="stop-signal57948", n_clicks=0), style=center_style)
                    ])


labels_layout = html.Div([html.H1("Input your labels", id="title1", style=center_style),  
                        html.Div([
                        dcc.Input(id='input-on-submit', type='text', autoFocus=True),
                                html.Button('Add Label', id="new-label", n_clicks=0),
                                dcc.Link(html.Button("Submit labels list"), href="/annotate", refresh=True)
                                ], style=center_style),
                        html.Div(id='label-submit',
                                        children='Enter a label and press submit', style=center_style)])

## Index layout ##
app.layout = url_bar_and_content_div

## Complete layout ##
app.validation_layout = html.Div([
    url_bar_and_content_div,
    annotation_layout(),
    labels_layout])


## Flask routes ##
@server.route("/retrieve_query", methods=['POST'])
def retrieve_data():
    '''Retrieve the sorted unlabelled list of dicts of keys [filenames, scores]'''
    unlabelled_sorted_dict = json.loads(request.data)
    labeler.unlabelled = [os.path.split(x["filename"])[1] for x in unlabelled_sorted_dict]
    labeler.update_iter()
    print(f"Data retrieved, number of images to annotate remaining: {len(labeler.unlabelled)}")
    return ""


@server.route(f'{static_image_route}<image_name>')
def serve_image(image_name):
    if image_name not in labeler.unlabelled+labeler.test_set:
        raise Exception(f'"{image_name}" is excluded from the allowed static files')
    return flask.send_from_directory(config.IMAGE_DIRECTORY, image_name)

@server.route("/stop_annotating")
def serve_stop_image():
    return flask.send_file("picsell_logo.png")


## Dash callbacks ##

# Index callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if not labeler.labels_selected:
        return labels_layout
    else:
        if not labeler.trainer_inited:
            labeler.trainer_inited = True
            r = requests.post("http://localhost:3333/init_training", data=json.dumps({"labels_list": labeler.labels_list}))
        labeler.configure_labelmap()
        layout = annotation_layout()
        return layout
        
# Labels input
@app.callback(Output('label-submit', 'children'),
    [Input('new-label', 'n_clicks')],
    [State('input-on-submit', 'value')])
def form(n_clicks, value):
    if (not labeler.labels_selected and n_clicks>0):
            labeler.labels_list = labeler.labels_list[1:]
            labeler.labels_selected = True
    if value:
        if not value in labeler.labels_list:
            labeler.labels_list.append(value)
        msg = "  "
        for label in labeler.labels_list:
            msg+=" "+label+", "
        msg=msg[:-2]
        return f"The label list is:{msg}"
    else:
        return ""





# Show and annotate images
@app.callback(Output('image', 'src'),
     [Input({'role': 'label-button', 'index': ALL}, "n_clicks"), Input("stop-signal57948", "n_clicks")])
def update(*n_clicks):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0][:-9]
    if changed_id == "stop-signal57948":
        r = requests.post("http://localhost:3333/stop_training", data=json.dumps({}))
        return "/stop_annotating"           
    elif len(changed_id)>0:
        changed_id = changed_id.split(",")[0].split(":")[1].strip('"')
        

    if not labeler.test_set_done:
        try:
            image = str(next(labeler.test_set_iter))
        except StopIteration:
            labeler.test_set_done = True
        for label in labeler.labels_list:
            if label == changed_id:
                labeler.test_set_gt.append(labeler.labelmap[label])
                changed_id = None
                break
        if labeler.test_set_done:
            print("sending")
            path = os.path.join("labeler", config.IMAGE_DIRECTORY)
            to_send = [path+x for x in labeler.test_set]
            data = {"test_data": (to_send, labeler.test_set_gt), 
                    "labels_list": labeler.labels_list}
            test_queue.put(data)

    if labeler.test_set_done:
        try:
            image = str(next(labeler.iter_images))
            labeler.images_tosend.append(image)
            trigger = False
        except StopIteration:
            trigger = True
    
        for label in labeler.labels_list:
            if label == changed_id:
                labeler.ground_truths.append(labeler.labelmap[label])
                break
        if trigger:
            r = requests.post("http://localhost:3333/stop_training", data=json.dumps({}))
            return "/stop_annotating"

    if config.BUFFER_SIZE>len(labeler.ground_truths):
        return static_image_route + image
    else:
        data = labeler.prep_send_data()
        q_send.put(data)
        return static_image_route + image

    

if __name__ == '__main__':
    # app.run_server(debug=True, use_reloader=False) #to not launch twice everything
    app.run_server(debug=True, port=3334)
    