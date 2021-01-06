"""
Adapted from https://github.com/laridzhang/ASNet
"""

import numpy as np
import time
import torch

from src.utils import calculate_game, make_path
from src.crowd_count import CrowdCount
from src.data_multithread_preload import multithread_dataloader
from src.split import get_cutoff, get_area_masks
from src import network
import src.density_gen as dgen

input_width = 1024
input_height = 720

test_set_name = 'data_camera1'
test_flag = dict()
test_flag['preload'] = True
test_flag['label'] = False
test_flag['mask'] = False
test_flag['batch_size'] = 1
test_flag['img_size'] = (input_width, input_height)

test_model_path = r'./final_model/shtechA.h5'
test_data_config = dict()
test_data_config[test_set_name] = test_flag.copy()

# load data
all_data = multithread_dataloader(test_data_config)
data = all_data[test_set_name]['data']

# create model
net = CrowdCount(is_cuda=True)
network.load_net(test_model_path, net)

net.cuda()
net.eval()

total_forward_time = 0.0
save_path = './density_map_camera1/'
make_path(save_path)
count_result = []
count_result_split = []
img_list = []

# get partition masks
cutoff_points_path = './cutoff_points/camera1.txt'
cutoff_points, (H_split, V_split) = get_cutoff(cutoff_points_path, input_height, input_width)
# print(cutoff_points)
area_masks = get_area_masks(cutoff_points, H_split=H_split, V_split=V_split, input_height=input_height, input_width=input_width)
num_area_masks = len(area_masks)

# predicting
img_idx = -1
for blob in data:
    image_data = blob['image']
    for image_name in blob['image_name']:
        img_idx += 1
        img_list.append(image_name)

    start_time = time.perf_counter()
    estimate_map, _ = net(image_data)

    estimate_map = estimate_map.data.cpu().numpy()

    estimate_count = []
    estimate_count_split = []
    for x in estimate_map:
        areas_cnt = []
        for i in range(num_area_masks):
            areas_cnt.append(np.sum(area_masks[i] * x))
        estimate_count.append(sum(areas_cnt))
        estimate_count_split.append(areas_cnt)
    count_result += estimate_count
    count_result_split += estimate_count_split

    total_forward_time += time.perf_counter() - start_time

    for x in estimate_map:
        dgen.save_density_map(x.squeeze(), save_path, img_list[img_idx] + "_density_map.png")

print('Total forward time is %f seconds for %d samples.' % (total_forward_time, img_idx + 1))
for i, img in enumerate(img_list):
    # print('[%s] Crowd Count: %d' % (img, int(count_result[i]+0.5)))
    out_str = '[%02d] ' % (i+1)
    for j in range(num_area_masks):
        out_str += '[Area %d] %d  ' % (j+1, int(count_result_split[i][j]+0.5))
    print(out_str)


ground_truth_path = './mydata/camera1_count.txt'
result_path = 'camera1_result.txt'
strs_gt_cnt = []
mean_gt_cnts = [0 for i in range(num_area_masks)]
mean_abs_errors = [0 for i in range(num_area_masks)]
img_num = img_idx + 1

with open(ground_truth_path, 'r') as f:
    for i, line in enumerate(f):
        strs_gt_cnt.append(line)
        for j, cnt in enumerate(line.split()):
            cnt = int(cnt)
            mean_gt_cnts[j] += cnt
            mean_abs_errors[j] += abs(cnt - int(count_result_split[i][j]+0.5))

# accs = ['%.2f' % (1 - mean_abs_errors[i]/mean_gt_cnts[i]) for i in range(num_area_masks)]
mean_gt_cnts = ['%.2f' % (x/img_num) for x in mean_gt_cnts]
mean_abs_errors = ['%.2f' % (x/img_num) for x in mean_abs_errors]


with open(result_path, 'w') as out:
    out.write('Input scale(W*H): %d*%d\n' % (input_width, input_height))
    out.write('Total forward time is %f seconds for %d samples\n\n' % (total_forward_time, img_idx + 1))
    for i, array in enumerate(count_result_split):
        array = [str(int(x + 0.5)) for x in array]
        out.write(strs_gt_cnt[i])
        out.write(' '.join(array) + '\n\n')

    out.write('Mean absolute error     : ' + ' '.join(mean_abs_errors) + '\n')
    out.write('Mean ground truth count : ' + ' '.join(mean_gt_cnts) + '\n')
    # out.write('Accuracy                : ' + ' '.join(accs) + '\n')