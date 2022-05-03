import os
import cv2
import sys
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.ndimage import correlate
from scipy.ndimage.morphology import distance_transform_edt

import matplotlib.pyplot as plt
import pandas as pd

eps = sys.float_info.epsilon


def calculate_measures(gt_dir, sm_dir, measures, save=False, save_as_excel=False, beta=np.sqrt(0.3), gt_threshold=0.5):
    """
    function that calculates Saliency measures for given directories

    arameters
    ----------
    gt_dir : str
        The path to the ground truth directory
    sm_dir : str
        The path to the predicted saliency map directory
    measures : list
        list of measure names which need to be calculated
        supported measures: 'MAE'       => Mean Squared Error
                            'E-measure' =>  Enhanced-alignment measure
                            'S-measure' =>  Structure-measure
                            'Max-F'     =>  Maximum F-measure
                            'Adp-F'     =>  Adaptive F-measure
                            'Wgt-F'     =>  Weighted F-measure
    save : str
        If spesified, the results will be saved in 'save' directory
    beta : float
        beta parameter that is used in F-measure formula. default is sqrt(0.3)
    gt_threshold : float
        The threshold that is used to binrize ground truth maps.

    Returns
    -------
    values : dictionary
        a dict containing the results
    """

    values = dict()
    for idx in measures:
        values[idx] = list()
        if idx == 'Max-F':
            values['Precision'] = list()
            values['Recall']    = list()

    for gt_name in tqdm(glob(os.path.join(gt_dir, '*'))):
        _, name = os.path.split(gt_name)
        sm_name = os.path.join(sm_dir, name)

        if os.path.exists(sm_name):

            gt, sm = read_and_normalize(gt_name, sm_name, gt_threshold) # 调整到0~1（gt会根据0.5），调整为相同大小

            if 'MAE' in measures:
                values['MAE'].append(mean_square_error(gt, sm))
            if 'Adp-E-measure' in measures:
                values['Adp-E-measure'].append(adaptive_e_measure(gt, sm))
            if 'E-measure' in measures:
                values['E-measure'].append(e_measure(gt, sm))
            if 'S-measure' in measures:
                values['S-measure'].append(s_measure(gt, sm, gt_name))
            if 'Adp-F' in measures:
                values['Adp-F'].append(adaptive_fmeasure(gt, sm, beta))
            if 'Wgt-F' in measures:
                values['Wgt-F'].append(weighted_fmeasure(gt, sm))
            if 'Max-F' in measures: # 需要计算256个阈值，产生的二值saliency map
                prec, recall = compute_pre_rec(gt, sm)  # 256 thresholds between 0 and 1 , prec_recall(gt, sm, 256), compute_pre_rec(gt, sm)
                values['Precision'].append(prec) #
                values['Recall'].append(recall)

        else:
            print("\n{} not found!".format(os.path.basename(sm_name)))
            print('---' * 10)

    if 'MAE' in measures:
        values['MAE'] = np.mean(values['MAE'])

    if 'Adp-E-measure' in measures:
        values['Adp-E-measure'] = np.mean(values['Adp-E-measure'])
    if 'E-measure' in measures:
        values['E-measure'] = np.mean(values['E-measure'])
    if 'S-measure' in measures:
        values['S-measure'] = np.mean(values['S-measure'])

    if 'Adp-F' in measures:
        values['Adp-F'] = np.mean(values['Adp-F'])

    if 'Wgt-F' in measures:
        values['Wgt-F'] = np.mean(values['Wgt-F'])

    if 'Max-F' in measures:
        values['Precision'] = np.mean(np.hstack(values['Precision'][:]), 1)
        values['Recall'] = np.mean(np.hstack(values['Recall'][:]), 1)
        f_measures = (1 + beta ** 2) * values['Precision'] * values['Recall'] / (
                beta ** 2 * values['Precision'] + values['Recall'] + 1e-8)
        values['Fmeasure_all_thresholds'] = f_measures
        values['Max-F'] = np.max(f_measures)
        if 'Mean-F' in measures:
            values['Mean-F'] = np.mean(f_measures)

    if save:
        if not os.path.isdir(save):
            os.mkdir(save)
        for key in values.keys():
            np.save(os.path.join(save, key + ".npy"), values[key])

    if save_as_excel:
        pass

    return values


def read_and_normalize(gt_path, sm_path, gt_threshold=0.5):
    """
    function that reads, normalizes and crops a ground truth and a saliency map

    parameters
    ----------
    gt_path : str
        The path to a ground truth map
    sm_path : str
        The path to a predicted saliency map
    gt_threshold : float
        The threshold that is used to binrize ground truth maps.

    Returns
    -------
    gt_img, sm_img : numpy.ndarray
        The prepared arrays
    """
    gt_img = norm_img(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))
    gt_img = (gt_img >= gt_threshold).astype(np.float32)
    sm_img = norm_img(cv2.imread(sm_path, cv2.IMREAD_GRAYSCALE))
    if sm_img.shape[0] != gt_img.shape[0] or sm_img.shape[1] != gt_img.shape[1]:
        sm_img = cv2.resize(sm_img, (gt_img.shape[1], gt_img.shape[0]))

    return gt_img, sm_img


def norm_img(im):
    return cv2.normalize(im.astype('float'),
                         None,
                         0.0, 1.0,
                         cv2.NORM_MINMAX)
    # return im / (np.amax(im) + 1e-8)  # 注意不同的标准化方法会导致，mean-F 不一样


# MAE
def mean_square_error(gt, sm):
    return np.mean(np.abs(sm - gt))


# def e_measure(gt, sm, num=255):
#     mybins = np.arange(0, 256) / 255.0
#     # sm = np.expand_dims(sm, axis=0).repeat(255, axis=0) # 增广第0维度，并沿着0维度重复255遍
#     # mybins = np.expand_dims(mybins, axis=(1,2))*np.ones((255,3,3))
#
#     gtNum = gt[gt>0.5].size # 前景部分像素数目
#     bgNmu = gt[gt<=0.5].size # 背景部分像素数目
#     pix_num = gtNum + bgNmu  # 总像素数目
#     # 找到mask在gt的前景部分的像素区域
#     pp = sm[gt>0.5]
#
#     # 找到mask在背景部分的像素区域
#     nn = sm[gt<=0.5]
#
#
#
#     pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
#     nn_hist,nn_edges = np.histogram(nn,bins=mybins) # 256个数，255个段
#
#     # 这里是为了找到前景（从灰度值高的位置开始的）
#     pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
#     nn_hist_flip = np.flipud(nn_hist)
#
#     pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
#     nn_hist_flip_cum = np.cumsum(nn_hist_flip)
#
#     ## 计算y_mean
#     =np.histogram(sm, bins=mybins)
#     fm_list, gt_list = np.zeros(2,255), np.zeros(2, 255)
#
#
#     pred_mean = pp_hist_flip_cum/pix_num
#
#     fm_list[0,:] = 1 - pp_hist_flip_cum/pix_num
#     fm_list[1,:] = -sm_hist_flip_cum/pix_num
#
#
#     gt_list[0,:] = 1 - gt_hist_flip_cum/pix_num
#     gt_list[1,:] = -gt_hist_flip_cum/pix_num
#
#     align_list = 2 * gt_list * fm_list /(gt_list * gt_list + fm_list * fm_list + 1e-20)
#
#     enhanced = ((align_list + 1) **2) / 4
#
#     score_list = enhanced[0,:]*   pix_num / (pix_num - 1 + 1e-20)
#     return score_list

    # score = np.zeros(num)
    # thlist = np.linspace(0,1,num=255)
    # h,w = gt.shape
    # for i in range(num):
    #     sm = (sm >= thlist[i]).astype(np.float)
    #     fm = sm - sm.mean()
    #     gt = gt - gt.mean()
    #     align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
    #     enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
    #     score[i] = np.sum(enhanced) / (h*w - 1 + 1e-20)
    # return score



# E-measure
def adaptive_e_measure(gt, sm):  # 这是 adapt 自适应E-measure, 阈值为均值的2倍
    """
    This fucntion computes the Enhanced-alignment Measure (E-Measure) between the saliency map and the ground truth
    article: https://arxiv.org/abs/1805.10421
    original code [Matlab]: https://github.com/DengPingFan/E-measure

    parameters
    ----------
    gt : numpy.ndarray
        The path to the ground truth directory
    sm : numpy.ndarray
        The path to the predicted saliency map directory

    Returns
    -------
    value : float
        The calculated E-masure
    """
    sm = adptive_binary(sm)

    gt = gt.astype(np.bool)
    sm = sm.astype(np.bool)

    dgt = gt.astype(np.float32)
    dsm = sm.astype(np.float32)

    if np.sum(dgt) == 0:  # if the gt is completely black
        enhanced_matrix = 1.0 - dsm  # only calculate the black area of intersection
    elif np.mean(dgt) == 1:  # if the gt is completely white
        enhanced_matrix = dsm  # only calcualte the white area of intersection
    else:
        # Normal case:
        # 1.compute alignment matrix
        align_matrix = alignment_term(dsm, dgt)
        # 2.compute enhanced alignment matrix
        enhanced_matrix = enhanced_alignment_term(align_matrix)

    height, width = gt.shape
    value = np.sum(enhanced_matrix) / (height * width - 1 + eps)
    return value


def alignment_term(dgt, dsm):
    # compute global mean
    mu_fm = np.mean(dsm)
    mu_gt = np.mean(dgt)

    # compute the bias matrix
    align_fm = dsm - mu_fm
    align_gt = dgt - mu_gt

    # compute alignment matrix
    align_Matrix = 2 * (align_gt * align_fm) / (align_gt * align_gt + align_fm * align_fm + eps)
    return align_Matrix


def enhanced_alignment_term(align_matrix):
    enhanced = ((align_matrix + 1) ** 2) / 4
    return enhanced


def adptive_binary(sm):
    adaptive_threshold = 2 * np.mean(sm)

    if adaptive_threshold > 1:
        adaptive_threshold = 1

    binary_sm = (sm >= adaptive_threshold).astype(np.float32)

    return binary_sm


# S-Measure
# article: https://www.crcv.ucf.edu/papers/iccv17/1164.pdf
# Matlab code: https://github.com/DengPingFan/S-measure
def s_measure(gt, sm, gt_name):
    """
    This fucntion computes the structural similarity (S-Measure) between the saliency map and the ground truth
    article: https://www.crcv.ucf.edu/papers/iccv17/1164.pdf
    original code [Matlab]: https://github.com/DengPingFan/S-measure

    parameters
    ----------
    gt : numpy.ndarray
        The path to the ground truth directory
    sm : numpy.ndarray
        The path to the predicted saliency map directory

    Returns
    -------
    value : float
        The calculated S-masure
    """
    gt_mean = np.mean(gt)
    # print(gt_name, gt_mean)
    if gt_mean == 0:  # if the GT is completely black
        sm_mean = np.mean(sm)
        measure = 1.0 - sm_mean  # only calculate the area of intersection
        # print('------',gt_name, measure)
    elif gt_mean == 1:  # if the GT is completely white
        sm_mean = np.mean(sm)
        measure = sm_mean.copy()  # only calcualte the area of intersection
        # print('+++++++', gt_name, measure)
    else:
        gt[gt>0.5] = 1
        gt[gt<0.5] = 0

        alpha = 0.5
        measure = alpha * s_object(sm, gt) + (1 - alpha) * s_region(sm, gt) # s_region 可能为无穷大
        if measure < 0:
            measure = 0

    return measure


def ssim(gt, sm):
    gt = gt.astype(np.float32)

    height, width = sm.shape
    num_pixels = width * height

    # Compute the mean of SM,GT
    sm_mean = np.mean(sm)
    gt_mean = np.mean(gt)

    # Compute the variance of SM,GT
    sigma_x2 = np.sum(np.sum((sm - sm_mean) ** 2)) / (num_pixels - 1 + eps)
    sigma_y2 = np.sum(np.sum((gt - gt_mean) ** 2)) / (num_pixels - 1 + eps)

    # Compute the covariance
    sigma_xy = np.sum(np.sum((sm - sm_mean) * (gt - gt_mean))) / (num_pixels - 1 + eps)

    alpha = 4 * sm_mean * gt_mean * sigma_xy
    beta = (sm_mean ** 2 + gt_mean ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        ssim_value = alpha / (beta + eps)
    elif alpha == 0 and beta == 0:
        ssim_value = 1.0
    else:
        ssim_value = 0

    return ssim_value


def divide_sm(sm, x, y):
    # copy the 4 regions
    h, w = sm.shape #

    lt = sm[:y, :x]
    rt = sm[:y, x:w] # rt = sm[:y, x:]
    lb = sm[y:h, :x] #  lb = sm[y:, :x]
    rb = sm[y:h, x:w] # rb = sm[y:, x:]

    return lt, rt, lb, rb


def divide_gt(gt, x, y):
    height, width = gt.shape
    area = width * height

    # copy the 4 regions
    lt = gt[:y, :x]
    rt = gt[:y, x:width]  # rt = gt[:y, x:]
    lb = gt[y:height, :x]  # lb = gt[y:, :x]
    rb = gt[y:height, x:width]

    # The different weight (each block proportional to the GT foreground region).
    w1 = (x * y) / area
    w2 = ((width - x) * y) / area
    w3 = (x * (height - y)) / area
    w4 = 1.0 - w1 - w2 - w3

    return lt, rt, lb, rb, w1, w2, w3, w4


def centroid(gt):
    # col
    rows, cols = gt.shape

    if np.sum(gt) == 0:
        x = np.round(cols / 2)
        y = np.round(rows / 2)
        x = int(x)
        y = int(y)
        print('S-measure, The GT is pure black!',x ,y)
    else:
        total = np.sum(gt)
        i = np.arange(cols).reshape(1, cols) + 1
        j = np.arange(rows).reshape(rows, 1) + 1

        x = int(np.round(np.sum(np.sum(gt, 0, keepdims=True) * i) / total))
        y = int(np.round(np.sum(np.sum(gt, 1, keepdims=True) * j) / total))

    return x, y


def s_region(sm, gt): #(gt, sm)
    x, y = centroid(gt)
    gt_1, gt_2, gt_3, gt_4, w1, w2, w3, w4 = divide_gt(gt, x, y)

    sm_1, sm_2, sm_3, sm_4 = divide_sm(sm, x, y)

    q1 = ssim(sm_1, gt_1)
    q2 = ssim(sm_2, gt_2)
    q3 = ssim(sm_3, gt_3)
    q4 = ssim(sm_4, gt_4)

    region_value = w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    return region_value


def object(sm, gt): # (gt, sm)
    x = np.mean(sm[gt == 1])
    # compute the standard deviations of the foreground or background in sm
    sigma_x = np.std(sm[gt == 1])
    score = 2.0 * x / (x ** 2 + 1.0 + sigma_x + eps)
    return score


def s_object(sm, gt): # 以前写的：s_object(sm, gt)
    # compute the similarity of the foreground in the object level

    sm_fg = sm.copy()
    sm_fg[gt == 0] = 0
    o_fg = object(sm_fg, gt)

    # compute the similarity of the background
    sm_bg = 1.0 - sm.copy()
    sm_bg[gt == 1] = 0
    o_bg = object(sm_bg, gt==0) # gt == 0

    u = np.mean(gt)
    object_value = u * o_fg + (1 - u) * o_bg
    return object_value



# Weighted F-Measure
# article: https://ieeexplore.ieee.org/document/6909433
# Matlab code: https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/FGEval/
def weighted_fmeasure(gt, sm, beta2=1):
    """
    This fucntion computes Weighted F-Measure between the saliency map and the ground truth
    article: https://ieeexplore.ieee.org/document/6909433
    original code [Matlab]: https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/FGEval/

    parameters
    ----------
    gt : numpy.ndarray
        The path to the ground truth directory
    sm : numpy.ndarray
        The path to the predicted saliency map directory

    Returns
    -------
    value : float
        The calculated Weighted F-Measure
    """
    dst, idx = distance_transform_edt(1 - gt, return_indices=True)

    raw_idx = idx[0][gt == 0]
    col_idx = idx[1][gt == 0]

    e = np.abs(sm - gt).astype(np.float32)
    et = np.abs(sm - gt).astype(np.float32)

    et[gt == 0] = et[raw_idx, col_idx]

    k = matlab_style_gauss2d(shape=(7, 7), sigma=5)

    ea = correlate(et.astype(np.float32), k, mode='constant')
    min_e_ea = np.abs(sm - gt).astype(np.float32)

    min_e_ea[gt * (ea < e) == 1] = ea[gt * (ea < e) == 1]

    b = np.ones_like(gt).astype(np.float32)
    b[gt == 0] = 2 - 1 * np.exp(np.log(1 - 0.5) / 5. * dst[gt == 0])

    ew = min_e_ea * b
    tpw = np.sum(gt) - np.sum(ew[gt == 1])
    fpw = np.sum(ew[gt == 0])

    rec = 1 - np.mean(ew[gt == 1])  # Weighed Recall
    prec = tpw / (eps + tpw + fpw)  # Weighted Precision

    value = (1 + beta2) * (rec * prec) / (eps + (beta2 * rec) + prec)
    return value

def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h



# Adaptive F-measure

def adaptive_fmeasure(gt, sm, beta):
    """
    This fucntion computes Adaptive F-measure between the saliency map and the ground truth using
    the binary method proposed in:
    https://ieeexplore.ieee.org/document/5206596

    parameters
    ----------
    gt : numpy.ndarray
        The path to the ground truth directory
    sm : numpy.ndarray
        The path to the predicted saliency map directory

    Returns
    -------
    value : float
        The calculated Adaptive F-measure
    """
    gt_idx = np.where(gt > 0)
    gt_cnt = np.sum(gt)

    if gt_cnt == 0:
        # prec = []
        # recall = []
        prec = 0
        recall = 0
    else:
        adaptive_threshold = 2 * np.mean(sm)
        if adaptive_threshold > 1:
            adaptive_threshold = 1
        sm_binary = (sm >= adaptive_threshold).astype(np.float32)
        hit_cnt = np.sum(sm_binary[gt_idx])
        alg_cnt = np.sum(sm_binary)

        if hit_cnt == 0:
            prec = 0
            recall = 0
        else:
            prec = hit_cnt / (alg_cnt + eps)
            recall = hit_cnt / (gt_cnt + eps) # 自己改动 recall = hit_cnt / gt_cnt
    value = (1 + beta ** 2) * prec * recall / ((beta ** 2 * prec + recall) + eps)
    return value

def compute_pre_rec(gt, sm, mybins=np.arange(0,256)/255.0):

    if(len(gt.shape)<2 or len(sm.shape)<2):
        print("ERROR: gt or mask is not matrix!")
        exit()
    if(len(gt.shape)>2): # convert to one channel
        gt = gt[:,:,0]
    if(len(sm.shape)>2): # convert to one channel
        sm = sm[:,:,0]
    # 找到gt中所有的显著性像素的数目(因为gt中只有：0和254.99999...)
    gtNum = gt[gt>0.5].size # pixel number of ground truth foreground regions
    # 找到mask在gt的前景部分的像素区域
    pp = sm[gt>0.5] # mask predicted pixel values in the ground truth foreground region
    # 找到mask在背景部分的像素区域
    nn = sm[gt<=0.5] # mask predicted pixel values in the ground truth bacground region

    pp_hist,pp_edges = np.histogram(pp,bins=mybins) #count pixel numbers with values in each interval [0,1),[1,2),...,[mybins[i],mybins[i+1]),...,[254,255)
    nn_hist,nn_edges = np.histogram(nn,bins=mybins) # 256个数，255个段

    # 这里是为了找到前景（从灰度值高的位置开始的）
    pp_hist_flip = np.flipud(pp_hist) # reverse the histogram to the following order: (255,254],...,(mybins[i+1],mybins[i]],...,(2,1],(1,0]
    nn_hist_flip = np.flipud(nn_hist)

    pp_hist_flip_cum = np.cumsum(pp_hist_flip) # accumulate the pixel number in intervals: (255,254],(255,253],...,(255,mybins[i]],...,(255,0]
    nn_hist_flip_cum = np.cumsum(nn_hist_flip)

    precision = pp_hist_flip_cum/(pp_hist_flip_cum + nn_hist_flip_cum+1e-8) #TP/(TP+FP)
    recall = pp_hist_flip_cum/(gtNum+1e-8) #TP/(TP+FN)

    precision[np.isnan(precision)]= 0.0
    recall[np.isnan(recall)] = 0.0

    return np.reshape(precision, (255,1)), np.reshape(recall, (255,1)) # len(recall)


def prec_recall(gt, sm, num_th):
    """
    This fucntion computes Adaptive F-measure between the saliency map and the ground truth using
    the binary method proposed in:
    https://ieeexplore.ieee.org/document/5206596
    The results of this dunction will be used to calculate Max-F measure and plot PR and F-Threshold Curves
    parameters
    ----------
    gt : numpy.ndarray
        The path to the ground truth directory
    sm : numpy.ndarray
        The path to the predicted saliency map directory
    num_th : interger
        The total number of thresholds between 0 and 1
    Returns
    -------
    prec, recall:  numpy.ndarray
        The calculated Precision and Recall (shape: (num_th,1))
    """
    gt_idx = np.where(gt > 0)
    gt_cnt = np.sum(gt)

    if gt_cnt == 0:
        # prec = []
        # recall = []
        prec = np.zeros((num_th, 1), np.float32)
        recall = np.zeros((num_th, 1), np.float32)

    else:
        hit_cnt = np.zeros((num_th, 1), np.float32)
        alg_cnt = np.zeros((num_th, 1), np.float32)
        thresholds = np.linspace(0, 1, num_th)
        for k, curTh in enumerate(thresholds):
            sm_binary = (sm >= curTh).astype(np.float32)
            hit_cnt[k] = np.sum(sm_binary[gt_idx])
            alg_cnt[k] = np.sum(sm_binary)

        prec = hit_cnt / (alg_cnt + eps)
        recall = hit_cnt / (gt_cnt + eps) #  recall = hit_cnt / gt_cnt

    return prec, recall


# Precision-Recall curves
def plot_save_pr_curves(PRE, REC, method_names, lineSylClr, linewidth, xrange=(0.0,1.0), yrange=(0.0,1.0), dataset_name = 'TEST', save_dir = './', save_fmt = 'pdf'):

    fig1 = plt.figure(1)
    num = PRE.shape[0]
    for i in range(0,num):
        if (len(np.array(PRE[i]).shape)!=0):
            plt.plot(REC[i], PRE[i],lineSylClr[i],linewidth=linewidth[i],label=method_names[i])

    plt.xlim(xrange[0],xrange[1])
    plt.ylim(yrange[0],yrange[1])

    xyrange1 = np.arange(xrange[0],xrange[1]+0.01,0.1)
    xyrange2 = np.arange(yrange[0],yrange[1]+0.01,0.1)

    plt.tick_params(direction='in')
    plt.xticks(xyrange1,fontsize=15,fontname='serif')
    plt.yticks(xyrange2,fontsize=15,fontname='serif')

    ## draw dataset name
    plt.text((xrange[0]+xrange[1])/2.0,yrange[0]+0.02,dataset_name,horizontalalignment='center',fontsize=20, fontname='serif',fontweight='bold')

    plt.xlabel('Recall',fontsize=20,fontname='serif')
    plt.ylabel('Precision',fontsize=20,fontname='serif')

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 11,        # 'size': 7
    }

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [len(handles)-x for x in range(1,len(handles)+1)]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='lower left', prop=font1)
    plt.grid(linestyle='--')
    fig1.savefig(save_dir+dataset_name+"_pr_curves."+save_fmt,bbox_inches='tight',dpi=300)
    print('>>PR-curves saved: %s'%(save_dir+dataset_name+"_pr_curves."+save_fmt))


# F-measure curves
def plot_save_fm_curves(FM, mybins, method_names, lineSylClr, linewidth, xrange=(0.0,1.0), yrange=(0.0,1.0), dataset_name = 'TEST', save_dir = './', save_fmt = 'pdf'):

    fig2 = plt.figure(2)
    num = FM.shape[0]
    for i in range(0,num):
        if (len(np.array(FM[i]).shape)!=0):
            plt.plot(np.array(mybins[0:-1]).astype(np.float)/255.0, FM[i],lineSylClr[i],linewidth=linewidth[i],label=method_names[i])

    plt.xlim(xrange[0],xrange[1])
    plt.ylim(yrange[0],yrange[1])

    xyrange1 = np.arange(xrange[0],xrange[1]+0.01,0.1)
    xyrange2 = np.arange(yrange[0],yrange[1]+0.01,0.1)

    plt.tick_params(direction='in')
    plt.xticks(xyrange1,fontsize=15,fontname='serif')
    plt.yticks(xyrange2,fontsize=15,fontname='serif')

    ## draw dataset name
    plt.text((xrange[0]+xrange[1])/2.0,yrange[0]+0.02,dataset_name,horizontalalignment='center',fontsize=20, fontname='serif',fontweight='bold')

    plt.xlabel('Thresholds',fontsize=20,fontname='serif')
    plt.ylabel('F-measure',fontsize=20,fontname='serif')

    font1 = {'family': 'serif',
    'weight': 'normal',
    'size': 11,
    }

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [len(handles)-x for x in range(1,len(handles)+1)]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],loc='lower left', prop=font1)
    plt.grid(linestyle='--')
    fig2.savefig(save_dir+dataset_name+"_fm_curves."+save_fmt,bbox_inches='tight',dpi=300)
    print('>>F-measure curves saved: %s'%(save_dir+dataset_name+"_fm_curves."+save_fmt))