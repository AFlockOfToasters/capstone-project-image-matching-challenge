import os
import pandas as pd
import cv2
import kornia as K

def get_scenes(directory):
    # Generates a list of all building folder names
    scenes = next(os.walk(directory))[1]
    scenes.sort()
    return scenes

def image_resize(img, height):
    width = int(img.shape[1] * (height/img.shape[0]))
    dim = (width, height)
    resized = cv2.resize(img, dim)
    
    return resized

def load_pairs(scenes, datadir):
    """
    load the image data from the correcsponding directory, together with the pairing metrics.
    Args:
        scenes:     List of scenes (folder names under /train)
        datadir:    String of the directory to source image data from
    Returns:
        pair: dataframe containing the image pairings
        calibration: dataframe contaaining the calibration data per image, including the corresponding scene
    """
    pair = pd.DataFrame()    
    calibration = pd.DataFrame()
    # path which contains all the categories of images
    j = 0
    for scene in scenes:
        print(f'loading category {j+1} of {len(scenes)}: {scene}')
        j += 1
        # read and concatenate pairing datasets. add the "scene" column.
        pairpath = os.path.join(datadir,scene,"pair_covisibility.csv")
        if pair.empty:
            pair = pd.read_csv(pairpath)
            pair["scene"] = str(scene)
        else:
            pairappend = pd.read_csv(pairpath)
            pairappend["scene"] = str(scene)
            pair = pd.concat([pair,pairappend],axis=0)
    return pair

def load_torch_image(imgpath, device):
    img = cv2.imread(imgpath)
    scale = 840 / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)
