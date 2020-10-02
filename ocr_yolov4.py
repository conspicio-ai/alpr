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
########## Initialize OCR tsv writer ##########

ocr_save_filename = 'tsv_files/IvLabs_OCR_Day3.tsv'
ocr_writer = ConverterTSV(ocr_save_filename,file_type='ocr')

#################### IMPORTING MODEL FOR DIGIT RECOGNITION ####################

cfg_v4_alpha = 'pytorch-YOLOv4/cfg/digit.cfg'
weight_v4_alpha = 'weights/ocr.weights'

m_alpha = DarknetYolov4(cfg_v4_alpha)
m_alpha.load_weights(weight_v4_alpha)
num_classes_alpha = m_alpha.num_classes
class_names_alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
print('Loading weights from %s... Done!' % (weight_v4_alpha))

if use_cuda:
	# m.cuda()
	m_alpha.cuda()
	# yolo_vehicle.cuda()
	
######### MAKE SURE IMAGES ARE IN `test_images` directory inside the current directory ##########

img_dir = 'SIH_hackathon/OCR_day3/Day3'
img_list = os.listdir(img_dir)
img_list.sort()
cv2.namedWindow('digit_on_plate', cv2.WINDOW_NORMAL)

for img_loc in img_list:
	if ('.jpg' in img_loc) or ('.png' in img_loc) or ('.jpeg' in img_loc):
		arranged_plate_temp = ''

		img = cv2.imread(os.path.join(img_dir,img_loc))
		cv2.imshow('Image', img)
		confidence = 0.5

		sized = cv2.resize(img, (m_alpha.width, m_alpha.height))
		sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
		boxes = do_detect(m_alpha, sized, confidence , 0.4, use_cuda)

		if len(boxes[0])>0:
			digit_on_plate, cls_conf_plate, coordinates_all, labels= plot_boxes_cv2(img, boxes[0],classes_to_detect=class_names_alpha,fontScale=0.4,thick=1, savename=False, class_names=class_names_alpha, color=(0,0,0))
			# print(digit_on_plate.shape)
			# digit_on_plate = padder(size_digit[0], size_digit[1], digit_on_plate)
						
			alphanumerics,x_c_list,y_c_list = alphanumeric_segemntor(img, boxes[0],class_names=class_names_alpha)

			## Sort plate on basis of x axis
			x_c_sort_idx = np.sort(np.argsort(x_c_list))
			# arranged_plate = ''
			char_list = []
			for count, idx in enumerate(x_c_sort_idx):
				detected_letter, digit_img = alphanumerics[idx][0], alphanumerics[idx][1]
				# cv2.imshow(f'{count}. It seems like {detected_letter}',digit_img) #SHOW INDIVIDUAL 
				char_list = char_list + [detected_letter]
				#arranged_plate = arranged_plate+detected_letter
			arranged_plate_temp = plate_to_string(x_c_list, y_c_list, char_list, line_thresh = 10)
			# print(arranged_plate_temp)
			if arranged_plate_temp[0] in ['0','1','2','3','4','5','6','7','8','9']:
				arranged_plate_temp = arranged_plate_temp[1:]

			cv2.imshow('digit_on_plate', digit_on_plate)
			print('The number Plate is: ', arranged_plate_temp, '\n')
		ocr_writer.put_ocr(img_loc, arranged_plate_temp)
		""" Add .tsv storing process here """
		if cv2.waitKey(1) & 0xff == ord('q'):
			break
			
cv2.destroyAllWindows()
