import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import model
import validation
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_pairs_and_cal(scenes, datadir):
    
    """
    load the image data from the corresponding directory, together with the pairing metrics.
    Args:
        scenes:     List of scenes (folder names under /train)
        datadir:    String of the directory to source image data from

    Returns:
        pair: dataframe containing the image pairings
        calibration: dataframe containing the calibration data per image, including the corresponding scene
        scalings: dataframe containing the scaling factors to transfer positional data of the translation vectors into real-world dimensions
    """
    pair = pd.DataFrame()    
    calibration = pd.DataFrame()
    scalings = pd.read_csv(os.path.join(datadir, "scaling_factors.csv"))
    # path which contains all the categories of images
    j = 0
    for scene in scenes:
        print(f'loading category {j+1} of {len(scenes)}: {scene}')
        j += 1

        # read and concatenate pairing datasets. add the "scene" column.
        pairpath = os.path.join(datadir,scene,"pair_covisibility.csv")
        if pair.empty:
            pair = pd.read_csv(pairpath)
        else:
            pairappend = pd.read_csv(pairpath)
            pair = pd.concat([pair,pairappend],axis=0)

        # read and concatenate calibration data, add the "scene" column.
        calibrationpath = os.path.join(datadir,scene,"calibration.csv")
        if calibration.empty:
            calibration = pd.read_csv(calibrationpath)
            calibration["scene"] = str(scene)
        else:
            calibrationappend = pd.read_csv(calibrationpath)
            calibrationappend["scene"] = str(scene)
            calibration = pd.concat([calibration,calibrationappend],axis=0)

    return pair, calibration, scalings



def plot_camera_positions(Rs, Ts, img_ids, scene="", scalings = pd.read_csv(os.path.join('../../data/train/', "scaling_factors.csv")), cmap="Blues", opacity = 1):
    scale = 1
    if scene != "":
        scale = scalings.query("scene == @scene")["scaling_factor"]
    C_x = []
    C_y = []
    C_z = []
    R_x = []
    R_y = []
    R_z = []
    for R,T in zip(Rs,Ts):
        position = np.dot(-R.T, T)
        rotation = R.T
        C_x.append(float(position[0]*scale))
        C_y.append(float(position[1]*scale))
        C_z.append(float(position[2]*scale))
        # R_x.append(np.arctan2(rotation[2][1], rotation[2][2])*-180/np.pi)
        R_y.append(np.arctan2(-rotation[2][0], np.sqrt((rotation[2][1]**2)+(rotation[2][2]**2))))
        # R_z.append(np.arctan2(rotation[1][0], rotation[0][0])*-180/np.pi)
    angles = R_y
    arrowscale = np.mean([max(C_x)-min(C_x),max(C_y)-min(C_y)])*0.1
    
    x_dir = [-float(arrowscale)*np.sin(angle) + x for angle, x in zip(angles, C_x)]
    y_dir = [-float(arrowscale)*np.cos(angle) + y for angle, y in zip(angles, C_z)]
    # Build figure
    fig = go.Figure()
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
    )
    hover_dicts = [
        {
                "hovertemplate": 
                "<b>ID: %{customdata}</b><br>X: %{x},<br>Y: %{y}<extra></extra>",
        },
        {
                "hoverinfo": 'skip',
        }
    ]
    for x_, y_, s, c, line_c, hover in [(C_x, C_z, 0, 'LightSkyBlue', 'blue', hover_dicts[0]), (x_dir, y_dir, 0, 'red', 'red', hover_dicts[1])]:
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x = x_,
                y = y_,
                customdata=img_ids,
                marker=dict(
                    size = s,
                    opacity = 0,
                    color = "LightSkyBlue",
                    line = dict(
                        color = "black",
                        width = 2,
                    ),
                    symbol = "circle-dot"
                ),
                showlegend = False,
                **hover
            )
        )
    if cmap == "Blues":
        col = iter(plt.cm.Blues(np.linspace(0.01, 0.99, len(C_x))))
    elif cmap == "Reds":
        col = iter(plt.cm.Reds(np.linspace(0.01, 0.99, len(C_x))))
    else:
        raise Exception("I'm sorry! Supported cmaps are 'Reds' and 'Blues'!")
    for x_, y_, x_d, y_d in zip(C_x, C_z, x_dir, y_dir):
        fig.add_annotation(
            x=x_,  # arrows' head
            y=y_,  # arrows' head
            ax=x_d,  # arrows' tail
            ay=y_d,  # arrows' tail
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',  # if you want only the arrow
            showarrow=True,
            opacity = opacity,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=3,
            arrowcolor=str(list(next(col))).replace("[","rgba(").replace("]",")")
            )
    fig.update_layout(title_text=scene.replace("_", " ").title())
    return fig



    # def load_image(imgpath):
    # img = cv2.imread(imgpath)
    # scale = 840 / max(img.shape[0], img.shape[1])
    # w = int(img.shape[1] * scale)
    # h = int(img.shape[0] * scale)
    # img = cv2.resize(img, (w, h))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # return img