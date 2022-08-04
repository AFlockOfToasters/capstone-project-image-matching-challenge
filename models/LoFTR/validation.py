# Copied from https://www.kaggle.com/code/tmyok1984/imc2022-validation-code/notebook

import csv
import numpy as np
import os
import pandas as pd

from collections import namedtuple

Gt = namedtuple('Gt', ['K', 'R', 'T'])
eps = 1e-15

def LoadCalibration(filename):
    '''Load calibration data (ground truth) from the csv file.'''

    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue

            camera_id = row[0]
            K = np.array([float(v) for v in row[1].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[3].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)

    return calib_dict

def DecomposeFundamentalMatrixWithIntrinsics(F, K1, K2):
    '''Decompose the fundamental matrix into R and T, given the intrinsics.'''

    # This fundamentally reimplements this function: https://github.com/opencv/opencv/blob/be38d4ea932bc3a0d06845ed1a2de84acc2a09de/modules/calib3d/src/five-point.cpp#L742
    # This is a pre-requisite of OpenCV's recoverPose: https://github.com/opencv/opencv/blob/be38d4ea932bc3a0d06845ed1a2de84acc2a09de/modules/calib3d/src/five-point.cpp#L559
    # Instead of the cheirality check with correspondences, we keep and evaluate the different hypotheses downstream, and pick the best one.
    # Note that our metric does not care about the sign of the translation vector, so we only need to evaluate the two rotation matrices.
    E = np.matmul(K2.T, np.matmul(F, K1))

    U, S, Vh = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vh) < 0:
        Vh *= -1

    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R_a = np.matmul(U, np.matmul(W, Vh))
    R_b = np.matmul(U, np.matmul(W.T, Vh))
    T = U[:, -1]

    return R_a, R_b, T          # returns the rotation matrices R_a and R_b of shape 3x3 and the translation vector T of shape 3x1

def QuaternionFromMatrix(matrix):
    '''Transform a rotation matrix into a quaternion.'''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
              [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
              [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
              [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


def ComputeErrorForOneExample(q_gt, T_gt, q, T, scale):
    '''Compute the error metric for a single example. The function returns two errors, over rotation and translation. 
    These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy.'''

    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t

def ComputeMaa(err_q, err_t, thresholds_q, thresholds_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''

    assert len(err_q) == len(err_t)

    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)

def EvaluateSubmission(predictions, input_dir, scaling_dict, thresholds_q, thresholds_t, output_dir=None):
    '''Evaluate a prediction file against the ground truth.

    Note that only the subset of entries in the prediction file will be evaluated.'''

    # Extract a list of scenes from the predictions file. Note that there is a single dataset, so we do not keep track of it.
    scenes = []
    for prediction in predictions.keys():
        dataset, scene, pair = prediction.split(';')
        if scene not in scenes:
            scenes += [scene]

    # Load the ground truth.
    calib_dict = {}
    for scene in scenes:
        calib_dict[scene] = LoadCalibration(os.path.join(input_dir, scene, "calibration.csv"))

    errors_dict_q = {scene: {} for scene in scenes}
    errors_dict_t = {scene: {} for scene in scenes}
    for prediction_key, F_predicted in predictions.items():
        dataset, scene, pair = prediction_key.split(';')
        image_id_1, image_id_2 = pair.split('-')

        K1, R1_gt, T1_gt = calib_dict[scene][image_id_1].K, calib_dict[scene][image_id_1].R, calib_dict[scene][image_id_1].T.reshape((3, 1))
        K2, R2_gt, T2_gt = calib_dict[scene][image_id_2].K, calib_dict[scene][image_id_2].R, calib_dict[scene][image_id_2].T.reshape((3, 1))

        R_pred_a, R_pred_b, T_pred = DecomposeFundamentalMatrixWithIntrinsics(F_predicted, K1, K2)
        q_pred_a = QuaternionFromMatrix(R_pred_a)
        q_pred_b = QuaternionFromMatrix(R_pred_b)

        dR_gt = np.dot(R2_gt, R1_gt.T)
        dT_gt = (T2_gt - np.dot(dR_gt, T1_gt)).flatten()
        q_gt = QuaternionFromMatrix(dR_gt)
        q_gt = q_gt / (np.linalg.norm(q_gt) + eps)

        # blah blah cheirality...
        err_q_a, err_t_a = ComputeErrorForOneExample(q_gt, dT_gt, q_pred_a, T_pred, scaling_dict[scene])
        err_q_b, err_t_b = ComputeErrorForOneExample(q_gt, dT_gt, q_pred_b, T_pred, scaling_dict[scene])
        assert err_t_a == err_t_b
        errors_dict_q[scene][pair] = min(err_q_a, err_q_b)
        errors_dict_t[scene][pair] = err_t_a

    # Aggregate the results by computing the final metric for each scene, and then averaging across all scenes.
    maa_per_scene = {}
    for scene in scenes:
        maa_per_scene[scene], _, _, _ = ComputeMaa(list(errors_dict_q[scene].values()), list(errors_dict_t[scene].values()), thresholds_q, thresholds_t)

    if output_dir is not None:
        scene_list = []
        pair_list = []
        maa_list = []
        for prediction_key, F_predicted in predictions.items():
            dataset, scene, pair = prediction_key.split(';')
            image_id_1, image_id_2 = pair.split('-')

            e_dict_q = errors_dict_q[scene][pair]
            e_dict_t = errors_dict_t[scene][pair]

            maa_pair, _, _, _ = ComputeMaa([e_dict_q], [e_dict_t], thresholds_q, thresholds_t)

            scene_list.append(scene)
            pair_list.append(pair)
            maa_list.append(maa_pair)

        df = pd.DataFrame({"scene":scene_list, "pair":pair_list, "maa":maa_list})
        outpur_csv = os.path.join(output_dir, "validation_all.csv")
        df.to_csv(outpur_csv, index=False)

    return np.mean(list(maa_per_scene.values())), maa_per_scene, errors_dict_q, errors_dict_t

def evaluate(input_dir, sample_id_list, fund_matrix_list):
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)

    # Load per-scene scaling factors.
    scaling_dict = {}
    with open(os.path.join(input_dir, f"scaling_factors.csv")) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            scaling_dict[row[0]] = float(row[1])

    predictions = {}
    for sample_id, fundamental_matrix in zip(sample_id_list, fund_matrix_list):
        predictions[sample_id] = np.array([float(v) for v in fundamental_matrix.split(' ')]).reshape([3, 3])

    maa, _, _, _ = EvaluateSubmission(
        predictions,
        input_dir,
        scaling_dict,
        thresholds_q,
        thresholds_t)

    return maa

def evaluate_single(input_dir, sample_id, fund_matrix):
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)

    # Load per-scene scaling factors.
    scaling_dict = {}
    with open(os.path.join(input_dir, f"scaling_factors.csv")) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue
            scaling_dict[row[0]] = float(row[1])

    predictions = {}
    predictions[sample_id] = np.array([float(v) for v in fund_matrix.split(' ')]).reshape([3, 3])

    maa, _, errors_dict_q, errors_dict_t = EvaluateSubmission(
        predictions,
        input_dir,
        scaling_dict,
        thresholds_q,
        thresholds_t)

    return maa, errors_dict_q, errors_dict_t