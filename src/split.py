"""
Code for getting partition masks given cut-off annotations

@Author YuXin He
@Date 2020/9/29 20:31

"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2

def get_cutoff(cutoff_points_path, input_height, input_width):
    arr = []
    with open(cutoff_points_path, 'r') as f:
        for line in f:
            arr += list(map(int, line.split()))

    ori_width, ori_height = arr[0:2]
    H_split, V_split = arr[2:4]
    arr_x = [x/ori_width * input_width / 8 for x in arr[4::2]]
    arr_y = [y/ori_height * input_height / 8 for y in arr[5::2]]
    cutoff_points = list(zip(arr_x, arr_y))
    return cutoff_points, (H_split, V_split)

def get_area_masks(cutoff_points, H_split, V_split, input_height, input_width, down_size_factor=8):
    num_areas = H_split * V_split
    out_height = input_height // down_size_factor
    out_width = input_width // down_size_factor

    # require the scale of density map to be even number
    if out_height % 2 or out_width % 2:
        print('Error: Invaild Image Size')
        return None

    # check whether the # of cut-off points is consistent with partition
    if len(cutoff_points) != (H_split - 1 + V_split - 1) * 2:
        print('Error: Cutoff_points Inconsistent with Area_split')
        return None

    area_masks = [[[0]*out_width for _ in range(out_height)] for _ in range(num_areas)]

    # Boolean array that indicate whether the slope of corresponding line is infinity e.g. doesn't exist
    is_inf_slope = [cutoff_points[i*2][0] == cutoff_points[i*2+1][0] for i in range(H_split-1)]

    # Array of slopes(unused if doesn't exist) for horizontal split lines
    a_H = [0 for i in range(H_split-1)]
    # Array of intercepts(used as x_split if slope doesn't exist) for horizontal split lines
    b_H = [0 for i in range(H_split-1)]

    for i in range(H_split-1):
        if not is_inf_slope[i]:
            a_H[i] = (cutoff_points[i*2][1] - cutoff_points[i*2+1][1]) / (cutoff_points[i*2][0] - cutoff_points[i*2+1][0])
            b_H[i] = cutoff_points[i*2][1] - a_H[i] * cutoff_points[i*2][0]
        else:
            b_H[i] = cutoff_points[i*2][0]
        # print(a_H[i], b_H[i])

    # Array of slopes for vertical split lines
    a_V = [0 for i in range(V_split-1)]
    # Array of intercepts for vertical split lines
    b_V = [0 for i in range(V_split-1)]

    for i in range(0, V_split-1):
        a_V[i] = (cutoff_points[(H_split-1+i) * 2][1] - cutoff_points[(H_split-1+i) * 2 + 1][1]) \
                 / (cutoff_points[(H_split-1+i) * 2][0] - cutoff_points[(H_split-1+i) * 2 + 1][0])
        b_V[i] = cutoff_points[(H_split-1+i) * 2][1] - a_V[i] * cutoff_points[(H_split-1+i) * 2][0]
        # print(a_V[i], b_V[i])

    for i in range(out_height):
        for j in range(out_width):
            col = H_split-1  # column index of (i,j), starts from 0, initiated as H_split-1
            row = V_split-1  # row index of (i,j), starts from 0, initiated as V_split-1

            # get column index of (i,j)
            for s in range(H_split-1):
                if (is_inf_slope[s] and j <= b_H[s]) or (not is_inf_slope[s] and j - i/a_H[s] + b_H[s]/a_H[s] <= 0):
                    col = s
                    break

            # get row index of (i,j)
            for t in range(V_split-1):
                if i - a_V[t] * j - b_V[t] <= 0:
                    row = t
                    break

            # calculate mask index
            mask_index = row * H_split + col

            area_masks[mask_index][i][j] = 1

    area_masks = np.array(area_masks)

    for i, mask in enumerate(area_masks):
        mask = mask * 255
        cv2.imwrite('./area_mask_%d.jpg' % (i+1), mask)

    return area_masks


