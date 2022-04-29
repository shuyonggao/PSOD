
'''
Non-salient object supression make gt and mask
'''
import os
import cv2
import numpy as np
import json


sal_path = "./DataStorage/DUTS-TR_transformer_crf"
filled_img_pseudo_path = "./DataStorage/filled_transformer_crf_gt" 
filled_mask_pseudo_path = './DataStorage/filled_transformer_crf_mask'

json_path = '/home/gaosy/DATA/Gao_DUTS_TR/json'

if not os.path.exists(filled_img_pseudo_path):
    os.mkdir(filled_img_pseudo_path)

file_list = os.listdir(sal_path)
file_list.sort()

##------------------------------make gt--------------------------------##
for file in file_list:

    if (os.path.isfile(os.path.join(sal_path, file))):
        edge_map_orig = cv2.imread(os.path.join(sal_path, file), 0).astype(np.float32)
        edge_map_orig[edge_map_orig < 250] = 0     # leave high-confidence foreground   
        edge_map_orig[edge_map_orig == 255] = 254  # empty position 255
        # plt.imshow(edge_map_orig)
        # plt.show()
        edge_map = edge_map_orig.copy()

        data = json.load(open(os.path.join(json_path, file.split('.')[0]+'.json')))

        fore_ground_points = []
        for point in data['shapes']:
            if point['label'] == 'foreground':
                fore_ground_points.append(point['points'][0]) 

        for i, point in enumerate(fore_ground_points):
            print(point)
            seed_point = (int(point[0]), int(point[1]))
            print(i, ' : ',  seed_point)


            if edge_map[seed_point[1],seed_point[0]] < 50:  #  discard undetected salient points, 
                continue
            
            # if edge_map[seed_point[1],seed_point[0]] < 50:  # use the initial pseudo-labels: employ edges and points to generate forground area
            #     pass

            mask = np.zeros([edge_map.shape[0] + 2, edge_map.shape[1] + 2], np.uint8)
            cv2.floodFill(edge_map, mask, seed_point, (255, 100, 100), (20, 20, 20), (50, 50, 50),
                          cv2.FLOODFILL_FIXED_RANGE) # fill 255
            # print(edge_map.max())

        edge_map[edge_map != 255] = 0  # filter out non-saliet object, highlight salient object manually annotated by annotators

        foreground_map = edge_map
        cv2.imwrite(os.path.join(filled_img_pseudo_path, file), foreground_map)


##------------------------------------make mask-----------------------------##

if not os.path.exists(filled_mask_pseudo_path):
    os.mkdir(filled_mask_pseudo_path)

for file in file_list:
    if "ILSVRC2012_test_00001446.png" in file:
        print('stop')

    if (os.path.isfile(os.path.join(sal_path, file))):
        edge_map_orig = cv2.imread(os.path.join(sal_path, file), 0).astype(np.float32)
        edge_map_orig[edge_map_orig > 5] = 255  # leave high-confidence background
        edge_map_orig[edge_map_orig == 255] = 254  
        # plt.imshow(edge_map_orig)
        # plt.show()
        edge_map = edge_map_orig.copy()

        data = json.load(open(os.path.join(json_path, file.split('.')[0] + '.json')))

        fore_ground_points = []
        for point in data['shapes']:
            if point['label'] == 'foreground':
                fore_ground_points.append(point['points'][0])  


        for i, point in enumerate(fore_ground_points):
            seed_point = (int(point[0]), int(point[1]))
            print(i, ' : ', seed_point)

            mask = np.zeros([edge_map.shape[0] + 2, edge_map.shape[1] + 2], np.uint8)

            cv2.floodFill(edge_map, mask, seed_point, (255, 100, 100), (20, 20, 20), (50, 50, 50),
                          cv2.FLOODFILL_FIXED_RANGE)  
            print(edge_map.max())

        edge_map[edge_map != 255] = 0  
        edge_map = 255-edge_map
        background_map = edge_map / 255
        kernel = np.ones((10, 10), np.uint8)  # 3 -> 30
        background_map = cv2.erode(background_map, kernel)
        background_map = background_map * 255

        cv2.imwrite(os.path.join(filled_mask_pseudo_path, file), background_map)  # background map，it will be replaced by mask map

# generate the mask
for file in file_list:
    filled_background = cv2.imread(os.path.join(filled_mask_pseudo_path, file), 0).astype(np.float32)
    filled_foreground = cv2.imread(os.path.join(filled_img_pseudo_path, file), 0).astype(np.float32)

    final_mask = filled_background + filled_foreground
    cv2.imwrite(os.path.join(filled_mask_pseudo_path, file), final_mask)

print('Finished NSS_2nd_GtMask')