import sys
sys.path.append('pytorch-YOLOv4-master')
from tool.darknet2pytorch import Darknet as DarknetYolov4
import argparse
import cv2,time
import numpy as np

from tool.plateprocessing import find_coordinates, plate_detect
from tool.utils import alphanumeric_segemntor,plot_boxes_cv2
from tool.torch_utils import *


def findstring(elements, threshold):
	elements.sort(key = lambda x: x[1])
	upper = ''
	lower = ''
	sd = 0
	if abs(elements[0][1] - elements[-1][1]) < threshold:
		print('Single Line Case')
		sd = 0
	else:
		print('Double Line Case')
		sd = 1
	if sd == 0:
		elements.sort(key = lambda x: x[0])
		for element in elements:
			upper = upper + element[2]
		return upper
	else:
		av = (elements[0][1] + elements[-1][1])/2
		elements.sort(key = lambda x: x[0])
		
		#print(av)
		#print(elements)
		for element in elements:
			#print(element[1])
			if element[1] < av:
				upper = upper + element[2]
			else:
				lower = lower + element[2]
		return upper + lower


def plate_to_string(x_c, y_c, line):
	olist = list(zip(x_c, y_c, line))
	olist.sort(key = lambda x:x[0])
	if len(olist) > 1:
		if olist[0][1] < olist[1][1]:
			x_1 = olist[1][0]
			y_1 = olist[1][1]
		else:
			x_1 = olist[0][0]
			y_1 = olist[0][1]
		if olist[-1][1] < olist[-2][1]:
			x_2 = olist[-2][0]
			y_2 = olist[-2][1]
		else:
			x_2 = olist[-1][0]
			y_2 = olist[-1][1]
		if x_2 - x_1 != 0:	
			theta = np.arctan((y_1 - y_2)/(x_2 - x_1))
			rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
			olistnew = rotate(olist, rot)
			olistnew.sort(key = lambda x: x[0])
			plate = findstring(olistnew, threshold = 3)
			print('Plate = ',plate)
			return plate
		else:
			return "N/A"
	else:
		return "N/A"

use_cuda = True
#################### PLATE ####################

cfg_v4 = 'pytorch-YOLOv4-master/cfg/yolo-obj.cfg'
weight_v4 = 'num_plate.weights'

m = DarknetYolov4(cfg_v4)
m.load_weights(weight_v4)
num_classes = m.num_classes
class_names = ['plate']
print('Loading weights from %s... Done!' % (weight_v4))

#################### DIGIT ####################

cfg_v4_alpha = 'pytorch-YOLOv4-master/cfg/digit.cfg'
weight_v4_alpha = 'yolo-obj_afterbeforefinal.weights'

m_alpha = DarknetYolov4(cfg_v4_alpha)
m_alpha.load_weights(weight_v4_alpha)
num_classes_alpha = m_alpha.num_classes
class_names_alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
print('Loading weights from %s... Done!' % (weight_v4_alpha))


if use_cuda:
	m.cuda()
	# m_alpha.cuda()


# cap = cv2.VideoCapture('C:/Users/rohit/Videos/1.mp4')
cap = cv2.VideoCapture('1.mp4')

cap.set(3, 1280)
cap.set(4, 720)

print("Starting Detection...")

while True:
	ret, img = cap.read()
	sized = cv2.resize(img, (m.width, m.height))
	sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

	start = time.time()
	boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
	result_img = plot_boxes_cv2(img, boxes[0],fontScale=0.5,thick=2, 
				savename=False, class_names=class_names_alpha)
	cv2.imshow('Yolo plate detection', result_img)

	if len(boxes[0]) > 0 :
		x1, y1, x2, y2 = find_coordinates(img, boxes[0])
		plate_bb = img[y1:y2,x1:x2]

		######### DETECT Digits ############

		sized = cv2.resize(plate_bb, (m_alpha.width, m_alpha.height))
		sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
		confidence = 0.4
		boxes = do_detect(m_alpha, sized, confidence , 0.6, False)
		finish = time.time()
		print('Predicted in %f seconds.' % (finish - start))
		digit_on_plate = plot_boxes_cv2(plate_bb, boxes[0],fontScale=0.5,thick=2, 
						savename=False, class_names=class_names_alpha, color=(0,0,0))

		cv2.imshow('digit_on_plate', digit_on_plate)
		alphanumerics,x_c_list,y_c_list = alphanumeric_segemntor(plate_bb, boxes[0],class_names=class_names_alpha)

		## Sort plate on basis of x axis
		x_c_sort_idx = np.sort(np.argsort(x_c_list))
		arranged_plate = ''
		char_list = []
		for count, idx in enumerate(x_c_sort_idx):
			detected_letter, digit_img = alphanumerics[idx][0], alphanumerics[idx][1]
			# cv2.imshow(f'{count}. It seems like {detected_letter}',digit_img) #SHOW INDIVIDUAL 
			char_list = char_list + [detected_letter]
			#arranged_plate = arranged_plate+detected_letter
		arranged_plate = plate_to_string(x_c_list, y_c_list, char_list)
		print('The number Plate is: ', arranged_plate)

	else:
		print("No plate detected!")

	key = 0xff & cv2.waitKey(1)
	if key == ord('q'):
		break

cv2.destroyAllWindows()
cap.release()
