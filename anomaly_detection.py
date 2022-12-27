import cv2
import numpy as np
from utils import *


config = {
    'save_path': "./result/",
    'image_path': "./datasets/leather/train/good/",
    'test_path': "./datasets/leather/test/cut/",
    'gt_path': "./datasets/leather/ground_truth/cut/",
    'nx': 30,  # I will define ny= nx. It will divide a image into nx*ny batches
    'method': 25,  # define the method of HLAC features, 35 dimensions or 25 dimensions
    'use_binary': False,


}


# get gray image, bgr image and binary image
train_dic = read_path(config['image_path'])
test_dic = read_path(config['test_path'])


train_hlac = hlac_list(
    train_dic, config['method'], config['nx'], config['use_binary'])
test_hlac = hlac_list(
    test_dic, config['method'], config['nx'], config['use_binary'])

train_hlac_np0 = np.array(train_hlac)
train_hlac_np = train_hlac_np0.swapaxes(0, 1)  # for pca analysis
test_hlac_np = np.array(test_hlac)

hlac_score = calculate_score(
    test_hlac_np, train_hlac_np, config['nx'])  # pca score

test_bgr_np = np.array(test_dic['image_bgr'])
# evaluation and output threshold by using F1 score
val = eval_val(test_bgr_np, hlac_score, config, config['nx'])

# save the result image
for n in range(np.shape(test_hlac_np)[0]):
    out = visualize(test_bgr_np[n], hlac_score[n], config['nx'], val)
    cv2.imwrite(config['save_path'] + str(n)+".png", out)
