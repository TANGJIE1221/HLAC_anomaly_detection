import numpy as np  

def haar_feature (img, mask_list):
    result_list = []
    for mask in mask_list :
        square = np.zeros((np.shape(mask)[0],))
        for n in range(np.shape(mask)[0]) :
            xn = np.shape(img)[1]
            yn = np.shape(img)[0]
            x1 = int(xn*mask[n][0])
            x2 = int(yn*mask[n][2])
            y1 = int(xn*mask[n][1])
            y2 = int(yn*mask[n][3])
            
            square[n] = np.sum(img[x1:x2,y1:y2])
            if mask[n][4] == -1:
                square[n] = -square[n]
        result_list.append(np.sum(square))
    haar_feature = np.array(result_list)
    return haar_feature