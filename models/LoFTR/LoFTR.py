import os
import argparse
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import numpy as np
import kornia as K
import kornia.feature as KF
import torch
import cv2
import warnings
warnings.filterwarnings("ignore")

# Run this script to match all possible image matches in a folder (given as filepath).

def get_pairs(filepath):
    pairs = [combo for combo in combinations(os.listdir(filepath), 2)]
    return pairs

def load_torch_image(imgpath, device):
    img = cv2.imread(imgpath)
    scale = 1120 / max(img.shape[0], img.shape[1])
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)

def get_loftr_results(pairs, filepath):
    # Determine if a GPU is available, otherwise use CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Initialize LoFTR and load the outdoor weights
    matcher = KF.LoFTR(pretrained='outdoor')
    matcher = matcher.to(device).eval()
    # Run LoFTR on the image pairs loaded previously. The output is a DataFrame containing all relevant data for each image pair analyzed.
    fund_matrix_list = []
    pair_list = []
    fund_matrix_eval = []
    pair_eval = []
    mkpts0_list = []
    mkpts1_list = []
    mconf_list = []

    for pair in tqdm(pairs):
        
        img_id0 = pair[0]
        img_id1 = pair[1]

        img0_pth = os.path.join(filepath, str(img_id0))
        img1_pth = os.path.join(filepath, str(img_id1))
        img0 = load_torch_image(img0_pth, device)
        img1 = load_torch_image(img1_pth, device)
        batch = {"image0": K.color.rgb_to_grayscale(img0), 
                "image1": K.color.rgb_to_grayscale(img1)}
        
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
            
        F = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.2, 0.99999, 50000)
        

        fund_matrix_list.append(F[0])
        pair_list.append(pair)
        fund_matrix_eval.append(" ".join(str(num) for num in F[0].flatten().tolist()))
        mkpts0_list.append(mkpts0)
        mkpts1_list.append(mkpts1)
        mconf_list.append(mconf)
        
    results = pd.DataFrame({'pair': pair_list, 'fund_matrix': fund_matrix_list, 
                            'mkpts0': mkpts0_list, 'mkpts1': mkpts1_list, 'mconf': mconf_list, 
                            'fund_matrix_eval': fund_matrix_eval}) 

    return results

def main(filepath):
    pairs = get_pairs(filepath)
    results = get_loftr_results(pairs, filepath)
    csv_name = filepath.rsplit('/', 3)[-1]
    results.to_csv(f'{csv_name}.csv',index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Where are the images located?')
    parser.add_argument('--filepath', metavar='path', required=True,
                        help='the path to the images')
    args = parser.parse_args()
    main(filepath=args.filepath)
