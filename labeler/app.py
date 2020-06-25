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
server = flask.Flask('app', root_path=os.getcwd())
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('app', server=server, external_stylesheets=external_stylesheets)

## Labeler class definition ##


class Labeler():
    def __init__(self, png_dir):
        self.png_dir = png_dir
        path = os.path.join(config.ANNOTATIONS_SAVE_PATH, "annotations.json")
        if not os.path.isfile(path):
            self.unlabelled = self.configure_dir(png_dir=self.png_dir)
            random.shuffle(self.unlabelled)
            self.test_set = self.unlabelled[:int(config.TEST_SET_FRAC*len(self.unlabelled))] 
            self.test_set_done = False
            self.test_set_iter = np.nditer([self.test_set])
            self.test_set_gt = []
            [self.unlabelled.remove(p) for p in self.test_set if p in self.unlabelled]
            self.labels_list = []
            
            self.labels_selected = False
            
        else:
            with open(path, 'r') as f:
                data_json = json.load(f)
            self.labels_list = self.check_existence(data_json.get("labels_list"))
            self.labels_selected = True
            self.unlabelled = self.check_existence(data_json.get("unlabelled"))
            if "labelled_data" in data_json:
                data = data_json.get("labelled_data")
                status = 400
                while not status==200:
                    r = requests.post(config.TRAINER_IP+"/train", data=json.dumps({"labelled_data": data, 
                                                                             "labels_list": self.labels_list,
                                                                             "unlabelled": self.unlabelled}))
                    status = r.status_code
            if "test_data" in data_json:
                self.test_set_done = True
                status = 400
                while not status==200:
                    r = requests.post(config.TRAINER_IP+"/test_data", data=json.dumps({"test_data": data_json["test_data"], 
                                                                        "labels_list": data_json["labels_list"]}))
                    status = r.status_code
        self.trainer_inited = False
        self.images_tosend = []
        self.ground_truths = []
        self.update_iter()

    def check_existence(self, data):
        if not data:
            data = []
            return data
        else:
            return data
        

    def configure_dir(self, png_dir):
        for file in os.listdir(png_dir):
            if not file.endswith((".png", ".jpg", ".PNG", ".JPG")):
                raise ValueError("Your PNG_DIR does not contain only png or jpg, please clean it")               
        print("Images directory checked")
        
        return [os.path.join(png_dir, x) for x in os.listdir(png_dir)]
    
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
        to_keep = self.images_tosend.pop()
        [self.unlabelled.remove(p) for p in self.images_tosend if p in self.unlabelled]
        print(f"Number of images to annotate remaining: {len(self.unlabelled)}")
        # to_send_1 = [config.IMAGE_DIRECTORY+x for x in self.images_tosend]
        # to_send_2 = [config.IMAGE_DIRECTORY+x for x in self.unlabelled]
        data = {"labelled_data": (self.images_tosend, self.ground_truths), "labels_list": self.labels_list, "unlabelled": self.unlabelled}
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
            r = requests.post(config.TRAINER_IP+"/train", data=json.dumps(data))
            print("Data sent to trainer")

class SendTestSet(Thread):
    def __init__(self, test_queue, daemon=True):
        Thread.__init__(self, daemon=daemon)
        self.test_queue = test_queue
    def run(self):
        data = self.test_queue.get()
        r = requests.post(config.TRAINER_IP+"/test_data", data=json.dumps(data))

## Object and threads init ##
q_send = queue.Queue()
test_queue = queue.Queue()
sender = Sender(q_send)
sender.start()
test_set_sender = SendTestSet(test_queue)
test_set_sender.start()
labeler = Labeler(png_dir=config.IMAGE_DIRECTORY)

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

## Finished layout ##
static_image_route = "/static/"
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
    print(f"Query received, following images to annotate re-ordered")
    return ""


@server.route(f'{static_image_route}/<image_name>')
def serve_image(image_name):
    im_path = os.path.join(config.IMAGE_DIRECTORY, image_name)
    if hasattr(labeler, "test_set"):
        paths = labeler.unlabelled+labeler.test_set
    else:
        paths = labeler.unlabelled

    if im_path not in paths:
        raise Exception(f'"{im_path}" is excluded from the allowed static files')
    return flask.send_file(im_path)

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
            r = requests.post(config.TRAINER_IP+"/init_training", data=json.dumps({"labels_list": labeler.labels_list}))
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
        r = requests.post(config.TRAINER_IP+"/stop_training", data=json.dumps({}))
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
            print("Sending test data")
            # to_send = [config.IMAGE_DIRECTORY+x for x in labeler.test_set]
            data = {"test_data": (labeler.test_set, labeler.test_set_gt), 
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
            r = requests.post(config.TRAINER_IP+"/stop_training", data=json.dumps({}))
            return "/stop_annotating"

    if config.BUFFER_SIZE>len(labeler.ground_truths):
        return static_image_route + os.path.split(image)[1]
    else:
        data = labeler.prep_send_data()
        q_send.put(data)
        return static_image_route + os.path.split(image)[1]

    

if __name__ == '__main__':
    # app.run_server(debug=True, use_reloader=False) #to not launch twice everything
    app.run_server(host= '0.0.0.0', debug=True, port=3334)
    