from distutils.log import error
import pandas as pd
import os
import io
import base64
import numpy as np
import plotly
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import dash
from dash import html,dcc,ctx
from dash.dependencies import Input,Output,State
from dash.exceptions import PreventUpdate
import viz_utilities as vu
import LoFTR_plotly as lp
import json


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]


################################################################################
# APP INITIALIZATION
################################################################################
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# this is needed by gunicorn command in procfile
server = app.server


################################################################################
# PLOTS
################################################################################
ALLSCENES = ["brandenburg_gate", "british_museum", "buckingham_palace",
 "colosseum_exterior", "grand_place_brussels", "lincoln_memorial_statue",
 "notre_dame_front_facade", "pantheon_exterior", "piazza_san_marco",
 "sacre_coeur", "sagrada_familia", "st_pauls_cathedral", "st_peters_square",
 "taj_mahal", "temple_nara_japan", "trevi_fountain"]


curscene="british_museum"
input_dir = '../../data/train/'
plot_dir = "Plots/"
pairings, cal, scalings = vu.load_pairs_and_cal(ALLSCENES,input_dir)


def plotter(scene):
    print(f"preparing {scene}")
    scenecal = cal.query(f"scene == '{scene}'")
    print("calculating Rs/Ts")
    Rs = [np.array(scenecal.iloc[i,2].split()).reshape(3,3).astype(float) for i in range(scenecal.shape[0])]
    Ts = [np.array(scenecal.iloc[i,3].split()).reshape(3,1).astype(float) for i in range(scenecal.shape[0])]
    print("plotting")
    fig = vu.plot_camera_positions(Rs,Ts,scene=scene,img_ids=scenecal["image_id"])
    print("done.")
    return fig

def read_from_json(scene):
    print(f"loading {os.path.join(plot_dir,f'{scene}.json')}")
    figure = plotly.io.read_json(os.path.join(plot_dir,f"{scene}.json"))
    return figure

def write_to_json(scene,figure):
    print(f"writing figure for scene: {scene}")
    figure.write_json(os.path.join(plot_dir,f"{scene}.json"))
    print("done.")

try: 
    print("attempting to read from existing files")
    figlist = [read_from_json(scene) for scene in ALLSCENES]
    print("Success!")
    figures = dict(zip(ALLSCENES,figlist))
except:
    print("Failure to load from existing! Would you like to recreate the files?")
    recreateflag = input("Type y to proceed: ")
    if recreateflag != ("y" or "Y"):
        raise Exception("Failed to load data.")
    figlist = [plotter(scene) for scene in ALLSCENES]
    print("writing new json files")
    figures = dict(zip(ALLSCENES,figlist))
    for scene, figure in zip(ALLSCENES,figlist):
        write_to_json(scene, figure)
# to do: dieter will eine progress bar

fig = figures[f"{curscene}"]


################################################################################
# LAYOUT
################################################################################
app.layout = html.Div(
    [
        html.H2(
            id="title",
            children="Dashboard prototype",
        ),
        html.H3(
            id="subtitle",
            children="Select a scene below, then two images to match against each other.",
        ),
        html.Div(children="Choose a scene"),
        # dcc.Textarea(
        #     id="textarea-state-example",
        #     value="",
        #     style={"width": "100%", "height": 100},
        # ),
        dcc.Dropdown(
            id="dropdown-menu",
            options = ALLSCENES,
            value="british_museum"),
        dcc.Graph(id="scatterplot", figure=fig),
        html.Button("reset", id="resetbutton", n_clicks=0),
        html.Button("calculate", id="calculatebutton", n_clicks=0),
        html.Div(id = "selector", children="nothing selected"),
        html.Img(id="pairplot"),
        dcc.Markdown('''
        **To Do:**
        - Add plot interactivity to show images and image pair matchings
        - Add a second dashboard with upload functionality for own custom images, then calculate their relative translations and rotations",
        '''),
        dcc.Store(id="selectionbuffer", data=[])    # if this doesnt work, use pedantic!
    ]
)

################################################################################
# INTERACTION CALLBACKS
################################################################################
# https://dash.plotly.com/basic-callbacks

# dropdown callback for scene selection
@app.callback(
    Output("scatterplot", "figure"),
    Input("dropdown-menu", "value"))
def update_graph(value):
    print("plotting scene...")
    fig = figures[value]
    return fig

# selection callback for pair selection
@app.callback(
    Output("selector", "children"),
    Output("selectionbuffer", "data"),
    Input("scatterplot", "clickData"),
    Input("resetbutton", "n_clicks"),
    State("selectionbuffer","data"))
def return_clicked_id(clickData, reset, selections):
    if ctx.triggered_id == "resetbutton" and reset > 0:
        return ["nothing selected", []]
    sel = selections
    if not clickData:
        return ["nothing selected", []]
    clicked = str(clickData["points"][0]["customdata"])+".jpg"
    if clicked not in sel:
        sel.append(clicked)
    else:
        return [f"You can't select the same image twice. You selected {sel[0]} and {sel[1]}", sel]
    if selections == []:
        return ["nothing selected", sel]
    elif len(sel)==1:
        return [f"you selected {sel[0]}", sel]
    elif len(sel)>2:
        sel = sel[1:3]
    if len(sel)==2:
        return [f"you selected {sel[0]} and {sel[1]}", sel]
    else:
        return ["unexpected selection error", []]

@app.callback(
    Output("pairplot", "src"),
    Input("calculatebutton", "n_clicks"),
    State("dropdown-menu", "value"),
    State("selectionbuffer", "data")
)
def plot_imagepair(calculate, scene, selections):
    if calculate > 0:
        if len(selections) != 2:
            raise PreventUpdate
        print("loading plotpaths")
        imgpath1 = os.path.join(input_dir,scene,"images",selections[0])
        imgpath2 = os.path.join(input_dir,scene,"images",selections[1])
        print(imgpath1)
        print(imgpath2)
        print("buffering image")
        buf = lp.single_loftr_figure(imgpath1, imgpath2, alpha = 0.1, threshold = 0.9, lines = True, dpi = 150)
        print("encoding")
        imgdata = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
        print("done")
        return f"data:image/png;base64,{imgdata}"
    
# Add the server clause:
if __name__ == "__main__":
    app.run(debug=True)