import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn import metrics
from hlac_35 import hlac_35
from hlac_25 import hlac_25

# read image


def read_path(image_path):
    image_gray = []
    image_bgr = []
    image_bin = []
    path_list = os.listdir(image_path)
    path_list.sort()
    for filename in path_list:
        img = cv2.imread(image_path + filename)
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_bin = cv2.threshold(
            img_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] == 255
        image_bin.append(img_bin)
        image_gray.append(img_g)
        image_bgr.append(img)
    return {'image_gray': image_gray,
            'image_bgr': image_bgr,
            'image_bin': image_bin}


def split_into_batches(image, nx):
    ny = nx
    batches = []
    for y_batches in np.array_split(image, ny, axis=0):
        for x_batches in np.array_split(y_batches, nx, axis=1):
            batches.append(x_batches)
    return batches


def extract_batchwise(image, nx,  method):
    hlac_batches = []
    batches = split_into_batches(np.uint8(image), nx)
    for batch in batches:
        feature = np.zeros(method, dtype=np.uint64)
        if method == 25:
            result = hlac_25(batch, feature)
        else:
            result = hlac_35(batch, feature)
        hlac_batches.append(result)

    return np.array(hlac_batches)

# calculate the batches' probability of anomaly


def calculate_score(test_hlac_np, train_hlac_np, nx):
    ny = nx
    com = []
    pca = PCA(n_components=1)
    hlac_score = np.zeros((np.shape(test_hlac_np)[0], nx*ny))
    for i in range(nx*ny):
        # use pca to analyse the principal component
        pca.fit(train_hlac_np[i, :, :])
        com.append(pca.components_)
        for j in range(np.shape(test_hlac_np)[0]):
            test_hlac_np0 = test_hlac_np[j][i]
            test_hlac_np1 = np.expand_dims(test_hlac_np0, 0)
            # the distance between principal component and test_hlac
            hlac_score[j][i] = pca.transform(test_hlac_np1)

    # probability of anomaly
    hlac_score -= np.nanmin(hlac_score)
    hlac_score /= np.nanmax(hlac_score)

    return hlac_score  # shape: (amount of image,nx*ny)

# return the hlac features list for 35method or 25method


def hlac_list_cascade(dic, method, nX):
    n_channel = len(nX) * method
    cascade_result = []
    for i in dic['image_gray']:
        merge = np.zeros((nX[0] * nX[0], n_channel))
        hlac_result = []

        for nx in nX:
            hlac_result.append(extract_batchwise(
                i, nx,  method=method))
        sub_hlac_40 = hlac_result[0]
        sub_hlac_20 = hlac_result[1]
        sub_hlac_10 = hlac_result[2]

        for n_40 in range(nX[0]*nX[0]):
            yy_40 = n_40//nX[0]
            xx_40 = n_40 % nX[0]
            yy_20 = yy_40//2
            xx_20 = xx_40//2
            yy_10 = yy_20//2
            xx_10 = xx_20//2
            n_20 = int(yy_20 * 20 + xx_20)
            n_10 = int(yy_10 * 10 + xx_10)

            # 連結
            merge[n_40] = np.concatenate(
                [sub_hlac_40[n_40], sub_hlac_20[n_20], sub_hlac_10[n_10]])

        cascade_result.append(merge)
    # shape: (amount of images, nX[0]*nX[0], method*len(nX))
    return cascade_result


def hlac_list(dic, method, nx, use_binary=False):
    hlac_result = []
    if use_binary:
        for i in dic['image_bin']:
            hlac_result.append(extract_batchwise(
                i, nx,  method=method))

    else:
        for i in dic['image_gray']:
            hlac_result.append(extract_batchwise(
                i, nx, method=method))

    return hlac_result  # shape: (amount of images, nx*ny, method)


# return the result image
def visualize(image, prob, nx, val):
    ny = nx
    batches = split_into_batches(image, nx)
    dst = np.zeros_like(image)

    py = 0
    for y in range(ny):
        px = 0
        for x in range(nx):
            batch = batches[y * nx + x]
            angle = prob[y * nx + x]

            if angle >= val:  # threshold
                dst = cv2.rectangle(
                    dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, int(255 * prob[y * nx + x]), 0), -1)
                dst = cv2.rectangle(
                    dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, 255, 0), 1)

            px += batch.shape[1]
        py += batch.shape[0]

    return cv2.addWeighted(image, 0.2, dst, 0.8, 1.0)

# show batches' probability of anomaly as image


def image_score(image, hlac_score, nx):
    ny = nx
    batches = split_into_batches(image, nx)
    score = np.zeros((np.shape(image)[0], np.shape(image)[1]), dtype=float)

    py = 0
    for y in range(ny):
        px = 0
        for x in range(nx):
            batch = batches[y * nx + x]
            bin = hlac_score[y * nx + x]
            score[py:py + batch.shape[0], px:px + batch.shape[1]] = bin

            px += batch.shape[1]
        py += batch.shape[0]

    return score


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


# evaluation and output threshold by using F1 score
def eval_val(test_bgr_np, hlac_score, config, nx):
    gt_path = config['gt_path']
    save_path = config['save_path']
    gt_dic = read_path(gt_path)
    gt_np = np.array(gt_dic['image_gray'], dtype=float)

    gt_np[gt_np != 0] = 1.0  # gt mask

    score_list = []
    for n in range(np.shape(hlac_score)[0]):
        image_score_np = image_score(test_bgr_np[n], hlac_score[n], nx)
        plt.imsave(save_path+str(n)+"image_score.png", image_score_np)
        img = cv2.imread(save_path+str(n)+"image_score.png")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # show result by Otsu's method
        ret, binary = cv2.threshold(
            gray, 0.6, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(save_path+str(n)+"image_score_OTSU.png", binary)
        np.save(save_path+str(n)+"image_score.npy", image_score_np)

        score_list.append(image_score_np)

    img_score_np = np.array(score_list)  # each batch's probability of anomaly

    dic = compute_pixelwise_retrieval_metrics(img_score_np, gt_np)
    val = dic['optimal_threshold']  # threshold
    print(val)

    plt.cla()
    plt.plot(dic['fpr'], dic['tpr'])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC')
    plt.savefig(save_path+'roc.png')

    plt.cla()
    plt.plot(dic['recall'], dic['precision'])
    plt.axvline(val, color='k', ls='--')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precison/recall')
    plt.savefig(save_path+'recall.png')

    auroc = np.array(dic['auroc']).reshape(1, 1)
    optimal_fpr = np.array(dic['optimal_fpr']).reshape(1, 1)
    optimal_fnr = np.array(dic['optimal_fnr']).reshape(1, 1)
    optimal_threshold = np.array(dic['optimal_threshold']).reshape(1, 1)

    print(f'auroc:{auroc},optimal_fpr:{optimal_fpr},optimal_fnr:{optimal_fnr},optimal_threshold:{optimal_threshold}')
    np.savetxt(save_path + 'auroc.txt', auroc)
    np.savetxt(save_path + 'optimal_fpr.txt', optimal_fpr)
    np.savetxt(save_path + 'optimal_fnr.txt', optimal_fnr)
    np.savetxt(save_path + 'optimal_threshold.txt', optimal_threshold)

    return val


# use histogram to choose threshold

def visualize2(image, hlac_score, nx, save_path, n):
    ny = nx
    batches = split_into_batches(image, nx, ny)
    dst = np.zeros_like(image)
    hlac_score2 = hlac_score

    fig = plt.figure()

    # use use histogram and otsu method

    # val = filters.threshold_otsu(hlac_score)
    # plt.cla()
    # ax = fig.add_subplot(1,1,1)
    # ax.hist(hlac_score, bins=20)
    # plt.axvline(val, color='k', ls='--')
    # plt.savefig(save_path+'hist'+str(n)+'.png')

    plt.cla()
    plt.axis('off')

    hist2, bins2 = np.histogram(hlac_score2)

   # choose the largest value as threshold
    for i in range(len(bins2)-2, -1, -1):
        if hist2[i] != 0:
            val2 = bins2[i]
            break

    ax = fig.add_subplot(1, 1, 1)
    ax.hist(hlac_score2)
    plt.axvline(val2, color='k', ls='--')
    plt.savefig(save_path+'final_hist'+str(n)+'.png')

    score = np.zeros((np.shape(image)[0], np.shape(image)[1]), dtype=float)
    py = 0
    for y in range(ny):
        px = 0
        for x in range(nx):
            batch = batches[y * nx + x]
            angle = hlac_score2[y * nx + x]

            if angle >= val2:
                dst = cv2.rectangle(
                    dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, int(255 * hlac_score2[y * nx + x]), 0), -1)
                dst = cv2.rectangle(
                    dst, (px, py), (px + batch.shape[1], py + batch.shape[0]), (0, 255, 0), 1)
                score[py:py + batch.shape[0], px:px + batch.shape[1]] = angle
            px += batch.shape[1]
        py += batch.shape[0]
    return cv2.addWeighted(image, 0.2, dst, 0.8, 1.0), score
