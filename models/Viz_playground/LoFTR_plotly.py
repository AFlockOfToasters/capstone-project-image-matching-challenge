import io
from matplotlib import patheffects
import pandas as pd
import numpy as np
import kornia as K
import kornia.feature as KF
import torch
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import warnings
warnings.filterwarnings("ignore")

def load_image(imgpath, res = 840):
    img = cv2.imread(imgpath)
    scale = res / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_torch_image(imgpath, device, res=840):
    img = cv2.imread(imgpath)
    scale = res / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)

def single_loftr_figure(img0_pth, img1_pth, alpha = 1, threshold = 0, lines = True, dpi = 150, res=840, where="outdoor"):
    # Determine if a GPU is available, otherwise use CPU
    print("selecting device")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Initialize LoFTR and load the outdoor weights
    print("initializing")
    matcher = KF.LoFTR(pretrained=None)
    
    if where == "outdoor":
        matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    elif where == "indoor":
        matcher.load_state_dict(torch.load("weights/indoor_ds_new.ckpt")['state_dict'])
    else:
        raise Exception("No weights for LoFTR defined!")

    matcher = matcher.to(device).eval()
    # Run LoFTR
    print("loading images")
    img0_torch = load_torch_image(img0_pth, device, res)
    img1_torch = load_torch_image(img1_pth, device, res)
    batch = {"image0": K.color.rgb_to_grayscale(img0_torch), 
            "image1": K.color.rgb_to_grayscale(img1_torch)}
    print("matching images")
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    
    results = pd.DataFrame({'mkpts0': mkpts0.tolist(), 'mkpts1': mkpts1.tolist(), 'mconf': mconf.tolist()}) 
    print("loading images")
    img0 = load_image(img0_pth, res)
    img1 = load_image(img1_pth, res)

    color = cm.jet(results.query(f'mconf > {threshold}').mconf)
    text = [
    'LoFTR',
    'Matches: {}'.format(len(results.query(f'mconf > {threshold}')))]
    print("plotting images")
    plt.switch_backend('Agg')
    fig = plot_matches(img0, img1, np.array(results.query(f'mconf > {threshold}').mkpts0.values.tolist()), 
                                        np.array(results.query(f'mconf > {threshold}').mkpts1.values.tolist()), color, text, alpha, lines, dpi)
    print("initializing buf")
    buf = io.BytesIO() 
    print("saving buf")
    plt.savefig(buf, format = "png") # save to the above file object
    #print("closing pyplot")
    #plt.close()
    print("done")
    return buf

def plot_matches(
        img0, img1, mkpts0, mkpts1, color,
        text, alpha, lines, dpi):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    # draw matches
    if lines == True and mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1, alpha = alpha)
                                        for i in range(len(mkpts0))]
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txt
    txt = fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color="w")
    txt.set_path_effects([PathEffects.withStroke(linewidth =2, foreground="k")])
    
    return fig