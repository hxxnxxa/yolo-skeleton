import argparse
import glob
import cv2
import os
import math
import numpy as np


# Default data paths
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


# Translate the values
def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


# Calculate center point each bounding box
def bbox_center_calculator(img_path: str, lbl_path: str, save_path: str): 

    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path))

    img = glob.glob(os.path.join(img_path, '*.png'))

    for lbl in os.listdir(lbl_path):

        f = open(lbl_path,'r')
        data = f.read().splitlines()

        for line in data:
            lblList = []

            for point in data:
                lblList.append(point.split(' '))

            x_center_list = []
            y_center_list = []
            width_list = []
            height_list = []

        for i in range(len(lblList)):
            x_center_list.append(float(lblList[i][1]))
            y_center_list.append(float(lblList[i][2]))
            width_list.append(float(lblList[i][3]))
            height_list.append(float(lblList[i][4]))

        lblList_pascal = []


        # Append to a list to store the converted data
        for i in range(len(lblList)):
            lblList_pascal.append(yolo_to_pascal_voc(x_center_list[i], y_center_list[i], width_list[i], height_list[i], 256, 256))


        # Convert the yolo to the pascal_voc data type, we have to convert it to an integer type.
        # Declare the list to store coordinates
        x1_int = []
        y1_int = []
        x2_int = []
        y2_int = []


        # Append to the list
        for i in range(len(lblList)):
            x1_int.append(math.ceil(lblList_pascal[i][0]))
            y1_int.append(math.ceil(lblList_pascal[i][1]))
            x2_int.append(math.ceil(lblList_pascal[i][2]))
            y2_int.append(math.ceil(lblList_pascal[i][3]))


        # Draw circles (the length of lblList)
        for i in range(len(lblList)):
            cv2.circle(img[lbl], (math.ceil((((x1_int[i])+(x2_int[i]))/2)),math.ceil((((y1_int[i])+(y2_int[i]))/2))), 2, (0,0,255), -1)
            
        cv2.imwrite(save_path + lbl[:-4] + "_skeleton.png", img)
        img = np.full((256,256,3),255,np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, dest='img_path', help='Image file directory')
    parser.add_argument('--lbl-dir', type=str, dest='lbl_path', help='Label file directory')
    parser.add_argument('--output-dir', type=str, dest='save_path', help='Output directory to save generated')
    args = parser.parse_args()
    bbox_center_calculator(args.img_path, args.lbl_path, args.save_path)