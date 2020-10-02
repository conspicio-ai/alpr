import sys
sys.path.append('yolov3_detector')
from yolov3_custom_helper import yolo_detector
from darknet import Darknet
sys.path.append('pytorch-YOLOv4')
from tool.darknet2pytorch import Darknet as DarknetYolov4
import argparse
import cv2,time
import numpy as np
import time

from tool.plateprocessing import find_coordinates, plate_to_string, padder, get_color
from tool.utils import alphanumeric_segemntor,plot_boxes_cv2
from tool.torch_utils import do_detect

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

cap = cv2.VideoCapture('videos/gate_1.mp4')
cap.set(3, 1280)
cap.set(4, 720)

# cv2.namedWindow('vehicle', cv2.WINDOW_NORMAL)
print("Starting Detection...")
while True:	
	ret, img = cap.read()
	if not ret:
		break
	sized = cv2.resize(img, (m_vehicle.width, m_vehicle.height))
	sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

	confidence_vehicle = 0.2
	boxes = do_detect(m_vehicle, sized, confidence_vehicle, 0.6, use_cuda)
	result_img, cls_conf_plate, _, _ = plot_boxes_cv2(img, boxes[0],classes_to_detect=class_names_veh,fontScale=0.5,thick=2,savename=False)
	cls_conf_plate = float(cls_conf_plate)

	cv2.imshow('vehicle', result_img)
	key = 0xff & cv2.waitKey(1)
	if key == ord('q'):
		break

cv2.destroyAllWindows()


