import sys, os
sys.path.append('yolov3_detector')
from yolov3_custom_helper import yolo_detector
from darknet import Darknet
sys.path.append('pytorch-YOLOv4')
from tool.darknet2pytorch import Darknet as DarknetYolov4
import argparse
import cv2,time
import numpy as np
from tool.plateprocessing import find_coordinates, plate_to_string, padder, get_color
from tool.utils import alphanumeric_segemntor,plot_boxes_cv2
from tool.torch_utils import *
import time
from utility_codes.tsv_converter import ConverterTSV

use_cuda = True

#################### PLATE ####################

cfg_v4 = 'pytorch-YOLOv4/cfg/yolo-obj.cfg'
weight_v4 = 'weights/plate.weights'

m = DarknetYolov4(cfg_v4)
m.load_weights(weight_v4)
num_classes = m.num_classes
class_names = ['plate']
print('Loading weights from %s... Done!' % (weight_v4))

if use_cuda:
	m.cuda()
	# m_alpha.cuda()
	# yolo_vehicle.cuda()

vehicle_save_filename = 'tsv_files/plate_tester.tsv'
vehicle_writer = ConverterTSV(vehicle_save_filename,file_type='vehicle')

image_dir = 'SIH_hackathon/Detection_Day3/Day3'
image_files = os.listdir(image_dir)
image_files.sort()
OUTPUT_SIZE = (1280, 720)

for img_name in image_files:
	frame = cv2.imread(os.path.join(image_dir, img_name))

	h, w = frame.shape[0:2]
	sized = cv2.resize(frame, (m.width, m.height))
	sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
	confidence = 0.2

	boxes = do_detect(m, sized, confidence , 0.6, use_cuda)

	result_img, cls_conf_plate, coordinates_all, labels = plot_boxes_cv2(frame, boxes[0],classes_to_detect=class_names,fontScale=0.5,thick=2, savename=False, class_names=class_names)
	cls_conf_plate = float(cls_conf_plate)

	for i,co in enumerate(coordinates_all):
		print(co)
		data = [img_name, co, labels[i]]
		vehicle_writer.put_vehicle(img_name, co, 'plate')
		# vehicle_writer.put_vehicle(img_loc, c, 'plate')

		cv2.imshow('Image', result_img)
	
		if cv2.waitKey(1) & 0xff == ord('q'):
			break
# cv2.waitKey(0)
cv2.destroyAllWindows()

import pandas as pd
def merge_and_save(fp1, fp2, outfile_path):
    tsv_file1 = pd.read_csv(fp1, sep='\t', header=0)
    tsv_file2 = pd.read_csv(fp2, sep='\t', header=0)
    merged = pd.concat([tsv_file1, tsv_file2])
    outfile = merged.sort_values(by='Image').reset_index(drop=True)
    outfile.to_csv(outfile_path, sep='\t', index=False)
merge_and_save('tsv_files/plate_tester.tsv', 'tsv_files/vehicle_tester.tsv', 'tsv_files/IvLabs_Detection_Day3.tsv')

