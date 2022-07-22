import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input,Output,State
import viz_utilities as vu


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
ALLSCENES = ["british_museum", "brandenburg_gate", "buckingham_palace",
 "colosseum_exterior", "grand_place_brussels", "lincoln_memorial_statue",
 "notre_dame_front_facade", "pantheon_exterior", "piazza_san_marco",
 "sacre_coeur", "sagrada_familia", "st_pauls_cathedral", "st_peters_square",
 "taj_mahal", "temple_nara_japan", "trevi_fountain"]

scene="british_museum"
input_dir = '../../data/train/'
def plotter(scene):
    scene=[scene]
    pairings, cal, scalings = vu.load_pairs_and_cal(scene,input_dir)
    Rs = [np.array(cal.iloc[i,2].split()).reshape(3,3).astype(float) for i in range(cal.shape[0])]
    Ts = [np.array(cal.iloc[i,3].split()).reshape(3,1).astype(float) for i in range(cal.shape[0])]
    fig = vu.plot_camera_positions(Rs,Ts,scene=scene[0],img_ids=cal["image_id"])
    return fig, pairings, cal, scalings
fig, pairings, cal, scalings = plotter(scene)


################################################################################
# LAYOUT
################################################################################
app.layout = html.Div(
    [
        html.H2(
            id="title",
            children="Neuefische Interactive Dash Plotly Dashboard",
        ),
        html.H3(
            id="subtitle",
            children="Add some fish text and click, and the chart will change",
        ),
        html.Div(children="Choose a scene"),
        # dcc.Textarea(
        #     id="textarea-state-example",
        #     value="",
        #     style={"width": "100%", "height": 100},
        # ),
        dcc.Dropdown(
            id="dropdown-state-example",
            options = ["british_museum", "brandenburg_gate", "buckingham_palace",
                    "colosseum_exterior", "grand_place_brussels", "lincoln_memorial_statue",
                    "notre_dame_front_facade", "pantheon_exterior", "piazza_san_marco", 
                    "sacre_coeur", "sagrada_familia", "st_pauls_cathedral", "st_peters_square",
                    "taj_mahal", "temple_nara_japan", "trevi_fountain"],
            value="british_museum"),
        html.Button("Submit", id="textarea-state-example-button", n_clicks=0),
        html.Div(id="textarea-state-example-output", style={"whiteSpace": "pre-line"}),
        dcc.Graph(id="scatterplot", figure=fig),
    ]
)

################################################################################
# INTERACTION CALLBACKS
################################################################################
# https://dash.plotly.com/basic-callbacks
@app.callback(
    [
        Output("textarea-state-example-output", "children"),
        Output("scatterplot", "figure"),
    ],
    Input("textarea-state-example-button","n_clicks"),
    State("dropdown-state-example", "value"),
)
def update_output(n_clicks, value):
    fig, pairings, cal, scalings = plotter(value)
    text = "you said: " + value
    return text, fig


# Add the server clause:
if __name__ == "__main__":
    app.run_server()
