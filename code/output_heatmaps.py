'''
This code outputs the heatmaps from a list of pictures
'''

import json
import os
import tensorflow as tf
from models.nets import cpm_hand
import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import sys
import imageio

input_size = "368"
hmap_size = 46
cmap_radius = 21
joints = 21
stages = 6
kalman_noise = 3e-2

joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145],
                    [3, 123, 234],
                    [234, 45, 234],
                    [93, 82, 98],
                    [231, 3, 129],
                    [129, 122, 3],
                    [129, 212, 200]]

limbs = [[0, 1],
         [1, 2],
         [2, 3],
         [3, 4],
         [0, 5],
         [5, 6],
         [6, 7],
         [7, 8],
         [0, 9],
         [9, 10],
         [10, 11],
         [11, 12],
         [0, 13],
         [13, 14],
         [14, 15],
         [15, 16],
         [0, 17],
         [17, 18],
         [18, 19],
         [19, 20]
         ]

if sys.version_info.major == 3:
    PYTHON_VERSION = 3
else:
    PYTHON_VERSION = 2

def visualize_result(test_img, stage_heatmap_np):
    '''
    Returns 
    '''
    joint_coord_set = np.zeros((joints, 2))
    black_img = np.zeros_like(test_img)
    mask_img = np.zeros((test_img.shape[0], test_img.shape[1], 3))
    
    for joint_num in range(joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num % len(joint_color_code))
        joint_color = joint_color_code[color_code_num]
        cv2.circle(black_img, center=(joint_coord[1], joint_coord[0]), radius=4, color=joint_color, thickness=-1)
        cv2.circle(mask_img, center=(joint_coord[1], joint_coord[0]), radius=5, color=(255, 255, 255), thickness=-1)
    
    for limb_num in range(len(limbs)):
        x1 = joint_coord_set[limbs[limb_num][0], 0]
        y1 = joint_coord_set[limbs[limb_num][0], 1]
        x2 = joint_coord_set[limbs[limb_num][1], 0]
        y2 = joint_coord_set[limbs[limb_num][1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 5),
                                       int(deg),
                                       0, 360, 1)
            mask_polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 8),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = (joint_num % len(joint_color_code))
            limb_color = joint_color_code[color_code_num]

            cv2.fillConvexPoly(black_img, polygon, color=limb_color)
            cv2.fillConvexPoly(mask_img, mask_polygon, color = (255, 255, 255))

    masked_test_img = np.where(mask_img == 255, test_img, np.zeros_like(test_img))
    return black_img, masked_test_img
    
def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    images_folder = conf['images_folder']
    heatmaps_output_folder = conf['heatmaps_output_folder']
    masks_output_folder = conf['masks_output_folder']
    images_output_folder = conf['images_output_folder']
    model_path = conf['model_path']

    #1. Read list of images from folder
    images_ids = [a for a in os.listdir(images_folder) if a.endswith(".jpg")]

    #2. 
    tf_device = "/gpu:0"
    with tf.device(tf_device):
        input_data = tf.placeholder(dtype = tf.float32, shape = [None, input_size, input_size, 3],
                                        name = 'input_image')
        center_map = tf.placeholder(dtype = tf.float32, shape = [None, input_size, input_size, 1],
                                    name = 'center_map')
        model = cpm_hand.CPM_Model(stages, joints + 1)
        model.build_model(input_data, center_map, 1)
    
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model.load_weights_from_file(model_path, sess, False)
    
    test_center_map = cpm_utils.gaussian_img(input_size, input_size, input_size / 2, input_size / 2, cmap_radius)
    test_center_map = np.reshape(test_center_map, [1, input_size, input_size, 1])

    with tf.device(tf_device):
        for im_id in images_ids:
            im_path = os.path.join(images_folder, im_id)
            test_img = cpm_utils.read_image(im_path, [], input_size, 'IMAGE')
            test_img_resize = cv2.resize(test_img, (input_size, input_size))
            test_img_input = test_img_resize / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis = 0)
            predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap, model.stage_heatmap,
                                                            ],
                                                            feed_dict = {
                                                                'input_image:0': test_img_input,
                                                                'center_map:0': test_center_map
                                                            })
            last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:joints].reshape(
                (hmap_size, hmap_size, joints))
            last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
            demo_img, masked_img = visualize_result(test_img, last_heatmap)

            heatmap_path = os.path.join(heatmaps_output_folder, im_id + ".npy")
            mask_path = os.path.join(masks_output_folder, im_id)
            im_path = os.path.join(images_output_folder)
            np.save(heatmap_path, last_heatmap)
            imageio.imsave(mask_path, masked_img)
            imageio.imsave(im_path, demo_img)

if __name__ == "__main__":
    main()