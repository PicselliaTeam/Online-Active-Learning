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
        self.labels_list = ["a"]
        self.update_iter()
        self.labels_selected = False

    def configure_dir(self, png_dir):
        for file in os.listdir(png_dir):
            if not file.endswith((".png", ".jpg")):
                print("Your directory does not contain only images, please clean it :)")
                return
        print("Scanning {} ... OK".format(png_dir))
        return os.listdir(png_dir)
    
    def configure_labelmap(self):
        self.labelmap = {}
        for i, l in enumerate(self.labels_list):
            self.labelmap[l] = i

    def update_iter(self):
        [self.unlabelled.remove(p) for p in self.images_tosend if p in self.unlabelled]
        self.iter_images = np.nditer([self.unlabelled])
        print("iterator updated !")

    def prep_send_data(self):
        '''reqs = {"labelled_data": [impaths, labels] 
                "labels_list": self explanatory
                "unlabelled": [impaths] }'''      
        path = "./labeler/assets/images/"
        to_keep = self.images_tosend.pop()
        print("to keep", to_keep)
        print(len(self.unlabelled))
        [self.unlabelled.remove(p) for p in self.images_tosend if p in self.unlabelled]
        print(len(self.unlabelled))
        to_send_1 = [path+x for x in self.images_tosend]
        to_send_2 = [path+x for x in self.unlabelled]
        print(self.ground_truths, self.images_tosend)
        data = {"labelled_data": (to_send_1, self.ground_truths), "labels_list": self.labels_list, "unlabelled": to_send_2}
        self.ground_truths = []
        self.images_tosend = [to_keep]
        return data

class Sender(Thread):
    def __init__(self, q_out, daemon=True):
        Thread.__init__(self, daemon=daemon)
        self.q_send = q_send

    def run(self):
        while True:
            data = q_send.get()
            r = requests.post("http://localhost:3333/train", data=json.dumps(data))
            print("data sent!")


## Object and threads init ##
labeler = Labeler(png_dir=os.path.join("labeler", image_directory))
q_send = queue.Queue()
sender = Sender(q_send)
sender.start()

## Dash Layouts ##
static_image_route = "/static/"
center_style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}
image_style = {'display': 'flex', 'align-items': 'center', 'justify-content': 'center', "object-fit": "scale-down", "height":"90vh"}


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

layout_index = html.Div([
    dcc.Link('Navigate to "/labels_input"', href='/labels_input'),
    html.Br(),
    dcc.Link('Navigate to "/annotate"', href='/annotate'),
])


def annotation_layout():
    return html.Div([
                         
                    html.Div([html.Button(name1, id={'role': 'label-button', 'index': name1}, n_clicks=0) for k, name1 in enumerate(labeler.labels_list)], style=center_style),
                    html.Div([html.Img(id='image', style=image_style)], style=center_style)
                    ])


labels_layout = html.Div([html.H1("Input your labels", id="title1", style=center_style),
                        
                        html.Div([
                        dcc.Input(id='input-on-submit', type='text'),
                                html.Button('Add Label', id="new-label", n_clicks=0),
                                dcc.Link(html.Button("Submit labels list"), href="/annotate")
                                ], style=center_style),
                        html.Div(id='label-submit',
                                        children='Enter a label and press submit', style=center_style)
                    ])

## Index layout ##
app.layout = url_bar_and_content_div

## Complete layout ##
app.validation_layout = html.Div([
    url_bar_and_content_div,
    layout_index,
    annotation_layout(),
    labels_layout,
])


## Flask routes ##
@server.route("/retrieve_query", methods=['POST'])
def retrieve_data():
    '''Retrieve the sorted unlabelled list of dicts of keys [filenames, scores]'''
    unlabelled_sorted_dict = json.loads(request.data)
    labeler.unlabelled = [os.path.split(x["filename"])[1] for x in unlabelled_sorted_dict]
    labeler.update_iter()
    print("data retrieved!", len(labeler.unlabelled))
    return ""


@server.route(f'{static_image_route}<image_name>')
def serve_image(image_name):
    if image_name not in labeler.unlabelled:
        raise Exception(f'"{image_name}" is excluded from the allowed static files')
    return flask.send_from_directory(image_directory, image_name)

@app.server.route("/stop_annotating")
def serve_stop_image():
    return flask.send_file("picsell_logo.png")


## Dash callbacks ##

# Index callbacks
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname=="/labels_input":
        return labels_layout
    if pathname=="/annotate":
        if not labeler.labels_selected:
            labeler.labels_list = labeler.labels_list[1:]
            labeler.labels_selected = True
        r = requests.post("http://localhost:3333/init_training", data=json.dumps({"labels_list": labeler.labels_list}))
        labeler.configure_labelmap()
        layout = annotation_layout()
        return layout
    else:
        return layout_index

# Labels input
@app.callback(Output('label-submit', 'children'),
    [Input('new-label', 'n_clicks')],
    [State('input-on-submit', 'value')])
def form(n_clicks, value):
    if not labeler.labels_selected:
            labeler.labels_list = labeler.labels_list[1:]
            labeler.labels_selected = True
    if value:
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
     [Input({'role': 'label-button', 'index': ALL}, "n_clicks")])
def update(*n_clicks):
    try:
        image = str(next(labeler.iter_images))
        labeler.images_tosend.append(image)
    except StopIteration:
        return "/stop_annotating"

    msg = "No button was clicked"
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0][:-9]
    if len(changed_id)>0:
        changed_id = changed_id.split(",")[0].split(":")[1].strip('"')
    for label in labeler.labels_list:
        if label == changed_id:
            selected_label = label
            labeler.ground_truths.append(labeler.labelmap[selected_label])
            msg = f'Label selected was {selected_label}'
            break
    if BATCH_SIZE>len(labeler.ground_truths):
        print(labeler.ground_truths, labeler.images_tosend)
        return static_image_route + image

    else:
        data = labeler.prep_send_data()
        q_send.put(data)
        return static_image_route + image

    
BATCH_SIZE = 4

if __name__ == '__main__':
    # app.run_server(debug=True, use_reloader=False) #to not launch twice everything
    app.run_server(debug=True, port=3334)
    