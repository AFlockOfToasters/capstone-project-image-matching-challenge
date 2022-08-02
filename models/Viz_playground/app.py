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

# for tests with a smaller sample size of plots (replotting takes very long)
# ALLSCENES = ["brandenburg_gate", "british_museum"]

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

def imshow(idlist,scene=curscene,path=input_dir):
    imlist = []
    c = 1
    if len(idlist)>2:
        idlist = idlist[0:2]
    for id in idlist:
        impath = os.path.join(path,str(scene),"images",str(id))
        print("reading ",impath," as no. ",c)
        with open(impath,"rb") as openimage:
            img = base64.b64encode(openimage.read()).decode('utf-8')
        imlist.append(str(f"data:image/png;base64,{img}"))
        c += 1
    return imlist

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
    fig.update_xaxes(visible=False, showticklabels=False, showgrid = False, gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(visible=False, showticklabels=False, showgrid = False, gridcolor="rgba(0,0,0,0)")
    fig.update_traces(visible=False)
    fig.update_layout(margin=dict(l = 5, r = 5, t = 5, b = 5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
EMPTYFIGURE = empty_figure()
################################################################################
# LAYOUT
################################################################################
app.layout = html.Div(
    style={"height":"50%","width":"98%"},
    children=[
        # TITLE
        html.H3(children="TWO EYES SEE M👁RE"),  #"◉ 📷 (☉.☉) ◎ 👁 👀"
        html.H6(children="A Capstone-Project by Dr. Dieter Janzen and Dr. Bernd Ackermann"),
        # TOP QUADRANTS
        html.Div(children=[
            # TOP LEFT
            html.Div(children=[
                # UI
                html.Label("Scene"),
                dcc.Dropdown(id="dropdown-menu",placeholder="Choose a Scene",options = [{"label": scenetitle.replace("_", " ").title() ,"value": scenetitle} for scenetitle in ALLSCENES]),
                #html.Br(),
                html.Label("Threshhold"),
                dcc.Slider(min=0,max=1,id="threshhold_slider", value=0, tooltip={"placement":"bottom","always_visible":True}),
                #html.Br(),
                html.Label("Alpha"),
                dcc.Slider(min=0,max=1,id="alpha_slider", value=0.1, tooltip={"placement":"bottom","always_visible":True}),
                #html.Br(),
                html.Label("Image scale"),
                dcc.Slider(240,1200,40,id="scale_slider", value=840, marks={240:"240 (fast)",480:"",720:"",960:"",1200:"1200 (slow)"}, tooltip={"placement":"bottom","always_visible":True}),
                #html.Br(),
                html.Button("reset selection", id="resetbutton", n_clicks=0),
                html.Button("calculate matches", id="calculatebutton", n_clicks=0),
                # INTERACTIVE TOP-DOWN-VIEW PLOT
                html.Label("Click two views (arrows) in the plot below", style={"text-align":"center","display":"block"}),
                dcc.Graph(id="scatterplot", figure = EMPTYFIGURE, style={"height":"60%"}),
                ], 
                style = {"width":"40%"}
                ),
            # TOP RIGHT
            html.Div(children=[


                # TEXT SELECTION INDICATOR
                html.Label(id="selector", children="nothing selected",style={"font-weight":"bold","text-align":"center","display":"block"}),
                html.Div(children=[
                    html.Img(id="implot1", style={"width":"50%","padding":5, "object-fit":"contain"}),
                    html.Img(id="implot2", style={"width":"50%","padding":5, "object-fit":"contain"})
                    ],
                    style={"height":"30%",'display': 'flex', 'flex-direction': 'row'}),
                html.Img(id="pairplot", style={"width":"100%"})
                ],
                style = {"width":"60%"}
                )

        ], style={'display': 'flex', 'flex-direction': 'row'}#,"height":"50%"}
        ),
        # # BOTTOM QUADRANTS
        # html.Div([
        #     # BOTTOM LEFT
        #     html.Div(children=[
        #         # LOFTR-CALCULATED PAIRING-PLOT

        #         ],
        #         style={"width":"50%"}),
        #     # BOTTOM RIGHT
        #     html.Div(children=[ 
        #         # IMAGE SELECTION INDICATOR


        #     ],
        #     style={"width":"50%",'display': 'flex', 'flex-direction': 'row'})

        # ],
        # style={'display': 'flex', 'flex-direction': 'row'}),
        
        dcc.Markdown('''
        **To Do:**
        - Add a second dashboard with upload functionality for own custom images, then calculate their relative translations and rotations",
        '''),
        # INVISIBLE MEMORY STORAGE COMPONENT TO STORE SELECTION DATA
        dcc.Store(id="selectionbuffer", data=[])
    ]
)

################################################################################
# INTERACTION CALLBACKS
################################################################################
# https://dash.plotly.com/basic-callbacks

# dropdown callback for scene selection
@app.callback(
    Output("scatterplot", "figure"),
    Output("resetbutton","n_clicks"),
    Input("dropdown-menu", "value"),
    State("resetbutton","n_clicks"))
def update_graph(scene,reset):
    print("plotting scene...")
    if scene:
        fig = figures[scene]
        reset += 1
    else:
        raise PreventUpdate
    return fig, reset



# selection callback for pair selection
@app.callback(
    Output("selector", "children"),
    Output("selectionbuffer", "data"),
    Output("implot1", "src"),
    Output("implot2", "src"),
    Input("scatterplot", "clickData"),
    Input("resetbutton", "n_clicks"),
    State("selectionbuffer","data"),
    State("dropdown-menu", "value"))
def return_clicked_id(clickData, reset, selections, scene):
    if scene == None:
        raise PreventUpdate
    if ctx.triggered_id == "resetbutton" and reset > 0:
        print("resetting...")
        return ["nothing selected", [], "",""]
    sel = selections
    if not clickData:
        return ["nothing selected", [], "",""]
    clicked = str(clickData["points"][0]["customdata"])+".jpg"
    if clicked not in sel:
        sel.append(clicked)
    else:
        fig2 = imshow(sel,scene)
        return [f"You can't select the same image twice. You selected {sel[0]} and {sel[1]}", sel, fig2[0],fig2[1]]
    if selections == []:
        return ["nothing selected", [], "",""]
    elif len(sel)==1:
        fig2 = imshow(sel,scene)
        return [f"you selected {sel[0]}", sel, fig2[0],""]
    elif len(sel)>2:
        sel = sel[1:3]
    if len(sel)==2:
        fig2 = imshow(sel,scene)
        return [f"you selected {sel[0]} and {sel[1]}", sel, fig2[0],fig2[1]]
    else:
        print(sel)
        raise Exception("unexpected selection error")


@app.callback(
    Output("pairplot", "src"),
    Input("calculatebutton", "n_clicks"),
    State("dropdown-menu", "value"),
    State("selectionbuffer", "data"),
    State("threshhold_slider", "value"),
    State("alpha_slider", "value"),
    State("scale_slider", "value")
)
def plot_imagepair(calculate, scene, selections,threshhold,alpha,scale):
    if len(selections) != 2 or calculate == 0 or scene == None:
        raise PreventUpdate
    print("loading plotpaths")
    imgpath1 = os.path.join(input_dir,scene,"images",selections[0])
    imgpath2 = os.path.join(input_dir,scene,"images",selections[1])
    print(imgpath1)
    print(imgpath2)
    print("buffering image")
    buf = lp.single_loftr_figure(imgpath1, imgpath2, alpha = alpha, threshold = threshhold, lines = True, dpi = 150 , res=scale)
    print("encoding")
    imgdata = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    print("done")
    return f"data:image/png;base64,{imgdata}"
    
# Add the server clause:
if __name__ == "__main__":
    app.run(debug=True)