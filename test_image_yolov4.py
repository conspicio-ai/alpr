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

use_cuda = True

#################### Vehicle ####################
cfg_v4_veh = '/home/himanshu/pytorch-YOLOv4/cfg/yolov4.cfg'
weight_v4_veh = 'weights/yolov4.weights'

m_vehicle = DarknetYolov4(cfg_v4_veh)
m_vehicle.load_weights(weight_v4_veh)
num_classes = m_vehicle.num_classes

# class_names_veh = {'car':2,'motorbike':3,'bus':5,'truck':7}
class_names_veh = ['car','motorbike','bus','truck']
print('Loading weights from %s... Done!' % (weight_v4_veh))

if use_cuda:
	m_vehicle.cuda()
	# m_alpha.cuda()
	# yolo_vehicle.cuda()
	
print("Starting Detection...")

image_dir = 'SIH_hackathon/Detection_Day3/Day3'
image_files = os.listdir(image_dir)
image_files.sort()
OUTPUT_SIZE = (1280, 720)

class_names_veh = ['car','motorbike','bus','truck']

vehicle_save_filename = 'tsv_files/vehicle_tester.tsv'
vehicle_writer = ConverterTSV(vehicle_save_filename,file_type='vehicle')
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

for img_name in image_files:
	frame = cv2.imread(os.path.join(image_dir, img_name))#Give the frame here'
	# print(frame.shape)
	h, w = frame.shape[0:2]

	sized = cv2.resize(frame, (m_vehicle.width, m_vehicle.height))
	sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
	# print(m_vehicle.width, m_vehicle.height)
	
	# frame = cv2.resize(frame, OUTPUT_SIZE, interpolation = cv2.INTER_AREA)
	confidence_vehicle = 0.25
	boxes = do_detect(m_vehicle, sized, confidence_vehicle, 0.3, use_cuda)
	# print(boxesq[0])
	result_img, cls_conf_plate, coordinates_all, labels = plot_boxes_cv2(frame, boxes[0],classes_to_detect=class_names_veh,fontScale=0.5,thick=2,savename=False)
	cls_conf_plate = float(cls_conf_plate)

	for i,co in enumerate(coordinates_all):
		print(co)
		data = [img_name, co, labels[i]]
		vehicle_writer.put_vehicle(img_name, co, labels[i])

	cv2.imshow('Image', result_img)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
			
cv2.destroyAllWindows()

