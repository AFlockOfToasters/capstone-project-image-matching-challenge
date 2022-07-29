import io
import base64
import pandas as pd
import numpy as np
import kornia as K
import kornia.feature as KF
import torch
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def load_image(imgpath):
    img = cv2.imread(imgpath)
    scale = 840 / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_torch_image(imgpath, device):
    img = cv2.imread(imgpath)
    scale = 840 / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)

def single_loftr_figure(img0_pth, img1_pth, alpha = 1, threshold = 0, lines = True, dpi = 150):
    # Determine if a GPU is available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Initialize LoFTR and load the outdoor weights
    matcher = KF.LoFTR(pretrained='outdoor')
    matcher = matcher.to(device).eval()
    # Run LoFTR
    img0_torch = load_torch_image(img0_pth, device)
    img1_torch = load_torch_image(img1_pth, device)
    batch = {"image0": K.color.rgb_to_grayscale(img0_torch), 
            "image1": K.color.rgb_to_grayscale(img1_torch)}

    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()
    
    results = pd.DataFrame({'mkpts0': mkpts0.tolist(), 'mkpts1': mkpts1.tolist(), 'mconf': mconf.tolist()}) 
    
    img0 = load_image(img0_pth)
    img1 = load_image(img1_pth)

    color = cm.jet(results.query(f'mconf > {threshold}').mconf)
    text = [
    'LoFTR',
    'Matches: {}'.format(len(results.query(f'mconf > {threshold}')))]
    fig = plot_matches(img0, img1, np.array(results.query(f'mconf > {threshold}').mkpts0.values.tolist()), 
                                        np.array(results.query(f'mconf > {threshold}').mkpts1.values.tolist()), color, text, alpha, lines, dpi)
    buf = io.BytesIO() 
    plt.savefig(buf, format = "png") # save to the above file object
    plt.close()
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
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)
    
    return fig