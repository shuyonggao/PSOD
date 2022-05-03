'''
Adaptive flood filling

generate the initial pseudo labels

edge + point annotation -> gt & mask

'''
import os
import cv2
import numpy as np
import json


# edge_path = "/home/gaosy/DATA/Gao_DUTS_TR/edge/"
# filled_img_path = '/home/gaosy/DATA/Gao_DUTS_TR/filled_correct_img_gt_r4' # only foreground
# filled_mask_path = '/home/gaosy/DATA/Gao_DUTS_TR/filled_correct_mask_r4'
# json_path = '/home/gaosy/DATA/Gao_DUTS_TR/json'

edge_path = "./dataset/edge/"
filled_img_path = './dataset/filled_correct_img_gt' # only foreground
filled_mask_path = './dataset/filled_correct_mask'
json_path = './dataset/json'


if not os.path.exists(filled_img_path):
    os.mkdir(filled_img_path)

if not os.path.exists(filled_mask_path):
    os.mkdir(filled_mask_path)

file_list = sorted(os.listdir(edge_path))

for file in file_list:
    if (os.path.isfile(os.path.join(edge_path, file))):
        edge_map_orig = cv2.imread(os.path.join(edge_path, file))
        edge_map = edge_map_orig.copy()

        data = json.load(open(os.path.join(json_path, file.split('.')[0]+'.json')))

        ## apply adaptive flood filling to foreground to make initial pseudo gt
        fore_ground_points = []
        for point in data['shapes']:
            if point['label'] == 'foreground':
                fore_ground_points.append(point['points'][0]) # point['points'] is a list

        for i, point in enumerate(fore_ground_points):
            seed_point = (int(point[0]), int(point[1]))
            print(i, ' : ',  seed_point)

            mask = np.ones([edge_map.shape[0] + 2, edge_map.shape[1] + 2], np.uint8) * 255
            cv2.circle(mask, center=seed_point, radius=int(min(edge_map.shape[0], edge_map.shape[1]) / 5), color=0,
                       thickness=-1)  # seed_point: (column, row)

            cv2.floodFill(edge_map, mask, seed_point, (255, 100, 100), (20, 20, 20), (50, 50, 50),
                          cv2.FLOODFILL_FIXED_RANGE) # (255, 100, 100) filled color
            foreground_map = edge_map
            # print(edge_map.max())

        foreground_map = np.array(1-(edge_map == edge_map_orig).astype(np.int))*255
        cv2.imwrite(os.path.join(filled_img_path, file), foreground_map)

        ## make initial pseudo mask 
        back_ground_points = []
        for point in data['shapes']:
            if point['label'] == 'background':
                back_ground_points.append(point['points'][0]) 

        for i, point in enumerate(back_ground_points):
            seed_point = (int(point[0]), int(point[1]))  # (column, row)
            print(i, ' : ',  seed_point)

            mask = np.ones([edge_map.shape[0] + 2, edge_map.shape[1] + 2], np.uint8) * 255
            cv2.circle(mask, center=seed_point, radius=int(min(edge_map.shape[0], edge_map.shape[1]) / 5), color=0,
                       thickness=-1)

            cv2.floodFill(edge_map, mask, seed_point, (255, 100, 100), (20, 20, 20), (50, 50, 50),
                          cv2.FLOODFILL_FIXED_RANGE)  # edge_map is already filled by foreground point 

        filled_mask_map = np.array(1-(edge_map == edge_map_orig).astype(np.int))*255
        cv2.imwrite(os.path.join(filled_mask_path, file), filled_mask_map)

print('Finished EdgePoint2gt.py')
