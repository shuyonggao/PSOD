import os
import cv2
import numpy as np


'''
generate single channel map: 1: gt 2: bachground  255: unconcern area
'''

filled_mask_path = "/home/gaosy/DATA/Gao_DUTS_TR/filled_mask"
filled_img_gt_path = "/home/gaosy/DATA/Gao_DUTS_TR/filled_img_gt"

filled_gt_and_mask_path = "/home/gaosy/DATA/Gao_DUTS_TR/filled_gt_and_mask"

if not os.path.exists(filled_gt_and_mask_path ):
    os.mkdir(filled_gt_and_mask_path )

file_list = os.listdir(filled_mask_path)
file_list.sort()

for file in file_list:
    filled_mask = cv2.imread(os.path.join(filled_mask_path, file), 0).astype(np.float32)
    filled_gt = cv2.imread(os.path.join(filled_img_gt_path, file), 0).astype(np.float32)

    filled_mask[filled_mask <255] = 0
    filled_gt[filled_gt<255] = 0
    final_mask = filled_mask + filled_gt
    final_mask[final_mask>255] = 1 # foreground
    final_mask[final_mask==255] =2  # background
    final_mask[final_mask==0] = 255 # unconcern area
    cv2.imwrite(os.path.join(filled_gt_and_mask_path, file), final_mask)

print('Finished GtMaskOneMap.py')