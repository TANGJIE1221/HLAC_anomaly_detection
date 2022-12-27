import numpy as np
import random
from utils import *
from haar import *


def haar_mask_read(path):
    fp=open(path,'r')
    ls=[]
    mask=[]
    for line in fp:
        
        if line != '\n' :
            if line[0].isalpha() == True:
                continue
            else:
                
                line=line.strip('\n')  
                line0 = line.split(' ')
                line1 = filter(None, line0)
                line2 =list(map(float,line1))
                mask.append(line2)  
            
        else :
            mask_np = np.array(mask,dtype=float)
            mask=[]
            ls.append(mask_np)
    fp.close()
    print(f'the number of masks:{len(ls)}')
    return ls



def get_mask(config):
    all_mask_list = haar_mask_read(config['haar_mask_path'])
    mask_list =  random.sample (all_mask_list, config['n_mask']) 

    f = open(config['save_path'] + 'selected_mask.txt', 'w')
    for x in mask_list:
        f.write(str(x) + "\n")
    f.close()
    return mask_list


def extract_batchwise_haar(image, nx, mask_list):
    haar_batches = []
    batches = split_into_batches(np.uint8(image), nx)
   
    for batch in batches:
        result = haar_feature(batch,mask_list)
        haar_batches.append(result)
       
    return np.array(haar_batches)



def image_haar(image_list, config, mask_list):
    nX = config['nX']
    n_channel = len(nX) * config['n_mask']
    haar_list =[]

    for i in image_list :
        sub_haar =[]
        for nx in nX:
            sub_h = extract_batchwise_haar(i, nx, mask_list )
            sub_haar.append(sub_h)
            
        sub_haar_40 = sub_haar[0]
        sub_haar_20 = sub_haar[1]
        sub_haar_10 = sub_haar[2]
        
        merge = np.zeros((nX[0]*nX[0], n_channel))
        for n_40 in range(nX[0]*nX[0]):
            yy_40 = n_40//nX[0]
            xx_40 = n_40%nX[0]
            yy_20 = yy_40//2
            xx_20 = xx_40//2
            yy_10 = yy_20//2
            xx_10 = xx_20//2
            n_20 = int(yy_20 * 20 + xx_20)
            n_10 = int(yy_10 * 10 + xx_10)

            # 連結
            merge[n_40] = np.concatenate([sub_haar_40[n_40],sub_haar_20[n_20],sub_haar_10[n_10]])
        haar_list.append(merge)
    return haar_list