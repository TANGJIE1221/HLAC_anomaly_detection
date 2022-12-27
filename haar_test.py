import cv2
import numpy as np
from utils import *
from haar import *
from haar_feature_mask import *


config = {
            'save_path' :"./result/",
            'image_path' : "./datasets/leather/train/good/",
            'test_path' : "./datasets/leather/test/cut/",
            'gt_path' : "./datasets/leather/ground_truth/cut/",
            'nX' : [40,20,10] ,  #  I will define ny= nx. It will divide a image into nx*ny batches
            'haar_mask_path' : "./haar_mask.txt",
            'n_mask' : 8
            
}


mask_list = get_mask (config)


train_dic= read_path(config['image_path']) # get gray image, bgr image and binary image
test_dic = read_path(config['test_path'])

train_haar = image_haar(train_dic['image_gray'], config, mask_list)  
train_haar_np0 = np.array(train_haar)
print(f'shape of train hlac0:{np.shape(train_haar_np0)}') #(amount of images, nx*ny, len(nX)*n_mask)
train_haar_np = train_haar_np0.swapaxes(0,1) # for pca analysis

test_haar = image_haar(test_dic['image_gray'], config, mask_list) 
test_haar_np = np.array(test_haar)
print(f'shape of test hlac:{np.shape(test_haar_np)}')

haar_score = calculate_score(test_haar_np, train_haar_np, config['nX'][0]) #pca score

test_bgr_np = np.array(test_dic['image_bgr'])
val =eval_val(test_bgr_np, haar_score, config, config['nX'][0]) # evaluation and output threshold by using F1 score

# save the result image
for n in range(np.shape(test_haar_np)[0]) :        
        out = visualize(test_bgr_np[n], haar_score[n], config['nX'][0],val)
        cv2.imwrite(config['save_path'] + str(n)+".png",out)