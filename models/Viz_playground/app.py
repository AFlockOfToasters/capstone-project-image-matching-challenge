import io
import json
from distutils.log import error
import pandas as pd
import os
import base64
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html,dcc,ctx
from dash.dependencies import Input,Output,State
from dash.exceptions import PreventUpdate
import viz_utilities as vu
import LoFTR_plotly as lp
import cv2

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


def plotter(scene,ids=None):
    if ids:
        print("selected elements detected...")
        print(len(ids))
        cmap="Reds"
        scenecal = pd.DataFrame()
        for id in ids:
            scenecal.append(cal.query(f"image_id == {id}"))
    else:
        print(f"preparing {scene}")
        scenecal = cal.query(f"scene == '{scene}'")
        cmap="Blues"   
    print("calculating Rs/Ts")
    Rs = [np.array(scenecal.iloc[i,2].split()).reshape(3,3).astype(float) for i in range(scenecal.shape[0])]
    Ts = [np.array(scenecal.iloc[i,3].split()).reshape(3,1).astype(float) for i in range(scenecal.shape[0])]
    print("plotting")
    fig = vu.plot_camera_positions(Rs,Ts,scene=scene,img_ids=scenecal["image_id"],cmap=cmap)
    print("done.")
    return fig

def implot(idlist,scene=curscene,path=input_dir):
    fig = make_subplots(rows=1, cols=2)
    c = 1
    if len(idlist)>2:
        idlist = idlist[0:2]
    for id in idlist:
        impath = os.path.join(path,scene,"images",id)
        print("reading ",impath," as no. ",c)
        img = cv2.imread(impath)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print("scene was changed, stuff broke... 🤷")
        print("plotting")
        fig.add_trace(go.Image(z=img),1,c)
        print("done")
        c += 1
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
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
    print("Failure to load from existing files! Would you like to recreate the files?")
    recreateflag = input("Type y to proceed: ")
    if recreateflag != ("y" or "Y"):
        raise Exception("Failed to load data.")
    figlist = [plotter(scene) for scene in ALLSCENES]
    print("writing new json files")
    figures = dict(zip(ALLSCENES,figlist))
    for scene, figure in zip(ALLSCENES,figlist):
        write_to_json(scene, figure)
# to do: dieter will eine progress bar
def empty_figure():
    fig = go.Figure(data=go.Scatter(x=[],y=[]))
    fig.update_xaxes(visible=False)
fig1 = figures[f"{curscene}"]
fig2 = implot([],curscene)


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
        html.Div([
            html.Div(children=[
                html.Label("Choose a scene"),
                dcc.Dropdown(id="dropdown-menu",options = ALLSCENES),
                html.Br(),
                html.Label("Select two views (arrows) in the plot below"),
                dcc.Graph(id="scatterplot")
                ], 
                style = {"padding":10,"flex":1}
                ),
            html.Div(children=[
                html.Label("Threshhold"),
                dcc.Slider(0,1,0.2,id="threshhold_slider", value=0),
                html.Br(),
                html.Label("Alpha"),
                dcc.Slider(0,1,0.2,id="alpha_slider", value=1),
                html.Br(),
                html.Label("Image scale (WARNING: larger scales are more accurate, but need longer processing time!)"),
                dcc.Slider(240,1120,80,id="scale_slider", value=840),
                html.Br(),
                html.Button("reset selection", id="resetbutton", n_clicks=0),
                html.Button("calculate matches", id="calculatebutton", n_clicks=0),
                html.Br(),
                html.Label(id = "selector", children="nothing selected"),
                dcc.Graph(id="implot")
                ]
                ,style = {"padding":10,"flex":1}
                )
        ], style={'display': 'flex', 'flex-direction': 'row'}
        ),

        html.Img(id="pairplot", style={"width":"50%"}),
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
    if value:
        fig = figures[value]
    else:
        raise PreventUpdate
    return fig



# selection callback for pair selection
@app.callback(
    Output("selector", "children"),
    Output("selectionbuffer", "data"),
    Output("implot", "figure"),
    Input("scatterplot", "clickData"),
    Input("resetbutton", "n_clicks"),
    State("selectionbuffer","data"),
    State("dropdown-menu", "value"))
def return_clicked_id(clickData, reset, selections, scene):
    if ctx.triggered_id == "resetbutton" and reset > 0:
        print("resetting...")
        fig2 = implot([],scene)
        return ["nothing selected", [], fig2]
    sel = selections
    if not clickData:
        fig2 = implot([],scene)
        return ["nothing selected", [], fig2]
    clicked = str(clickData["points"][0]["customdata"])+".jpg"
    if clicked not in sel:
        sel.append(clicked)
    else:
        fig2 = implot(sel,scene)
        return [f"You can't select the same image twice. You selected {sel[0]} and {sel[1]}", sel, fig2]
    if selections == []:
        fig2 = implot([],scene)
        return ["nothing selected", sel, fig2]
    elif len(sel)==1:
        fig2 = implot(sel,scene)
        return [f"you selected {sel[0]}", sel, fig2]
    elif len(sel)>2:
        sel = sel[1:3]
    if len(sel)==2:
        fig2 = implot(sel,scene)
        return [f"you selected {sel[0]} and {sel[1]}", sel, fig2]
    else:
        print(sel)
        raise Exception("unexpected selection error")

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
        buf = lp.single_loftr_figure(imgpath1, imgpath2, alpha = 0.1, threshold = 0, lines = True, dpi = 150)
        print("encoding")
        imgdata = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
        print("done")
        return f"data:image/png;base64,{imgdata}"
    
# Add the server clause:
if __name__ == "__main__":
    app.run(debug=True)