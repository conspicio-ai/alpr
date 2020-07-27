import sys
sys.path.append('yolov3_detector')
from yolov3_custom_helper import yolo_detector
from darknet import Darknet
sys.path.append('pytorch-YOLOv4-master')
from tool.darknet2pytorch import Darknet as DarknetYolov4
import argparse
import cv2,time
import numpy as np
from tool.plateprocessing import find_coordinates, plate_to_string, padder, get_color
from tool.utils import alphanumeric_segemntor,plot_boxes_cv2
from tool.torch_utils import *



use_cuda = True
#################### PLATE ####################

cfg_v4 = 'pytorch-YOLOv4-master/cfg/yolo-obj.cfg'
weight_v4 = '/home/himanshu/Downloads/yolo-obj_last.weights'

m = DarknetYolov4(cfg_v4)
m.load_weights(weight_v4)
num_classes = m.num_classes
class_names = ['plate']
print('Loading weights from %s... Done!' % (weight_v4))

#################### DIGIT ####################

cfg_v4_alpha = 'pytorch-YOLOv4-master/cfg/digit.cfg'
weight_v4_alpha = '/home/himanshu/Downloads/alphanumeric.weights'

m_alpha = DarknetYolov4(cfg_v4_alpha)
m_alpha.load_weights(weight_v4_alpha)
num_classes_alpha = m_alpha.num_classes
class_names_alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
print('Loading weights from %s... Done!' % (weight_v4_alpha))

#################### VEHICLE ####################
cfgfile_yolov3 = "yolov3_detector/cfg/yolov3.cfg"
weightsfile_yolov3 = "yolov3_detector/yolov3.weights"
names_file_yolov3 = "yolov3_detector/data/coco.names"

yolo_vehicle = Darknet(cfgfile_yolov3)
yolo_vehicle.load_weights(weightsfile_yolov3)
yolo_vehicle.net_info["height"] = 160
yolo_vehicle.eval()

if use_cuda:
	m.cuda()
	m_alpha.cuda()
	yolo_vehicle.cuda()

############# READER/WRITER ##########	
size = (1280,720)
size_digit = (800,800)

cap = cv2.VideoCapture('/home/himanshu/sih_number_plate/3.mp4')
plate_1_writer = cv2.VideoWriter('plate_3.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 25, size) 
digit_1_writer = cv2.VideoWriter('digit_3.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, size_digit) 
cap.set(3, 1280)
cap.set(4, 720)

########## PUT TEXT #########
fontScale = 1
color = (0, 255, 0)  
thickness = 2
############################

########## MAJORITY AND INCREASE FPS ##########
plate_window = []
window_size = 5 # Mode of the list will be taken for these many samples
frame_add_interval = 2 # Only the second frame will be read
window_counter = 0
frame_counter = 0
started_counter = 0
############################

print("Starting Detection...")
while True:
	ret, img = cap.read()
	frame_counter = frame_counter + 1
	if not ret:
		break
	
	if frame_counter % frame_add_interval == 0:
		frame_counter = 0
		
		sized = cv2.resize(img, (m.width, m.height))
		sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

		start = time.time()

		img, vehicle_img, closest_vehicle_label = yolo_detector(img,use_cuda,yolo_vehicle,names_file_yolov3,INPUT_SIZE = (1280,720))
		if closest_vehicle_label is not None:
			print(closest_vehicle_label)
		print("Image shape after yolov3", img.shape)
		boxes = do_detect(m, sized, 0.2, 0.6, use_cuda)
		result_img, cls_conf_plate = plot_boxes_cv2(img, boxes[0],fontScale=0.5,thick=2, 
					savename=False, class_names=class_names)
		cls_conf_plate = float(cls_conf_plate)

		digit_on_plate = np.zeros((size_digit[0], size_digit[1], 3), dtype = np.uint8)
		# cv2.rectangle(result_img, (875, 0),(1280, 200),(0,0,0), thickness = -1)

		if len(boxes[0]) > 0 :

			x1, y1, x2, y2 = find_coordinates(img, boxes[0])
			plate_bb = img[y1:y2,x1:x2]
			print(plate_bb.shape)
			type_vehicle = get_color(plate_bb)

			######### DETECT Digits ############
			
			sized = cv2.resize(plate_bb, (m_alpha.width, m_alpha.height))
			sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
			confidence = 0.6
			boxes = do_detect(m_alpha, sized, confidence , 0.6, use_cuda)
		
			# print('Predicted in %f seconds.' % (FPS))
			digit_on_plate, _ = plot_boxes_cv2(plate_bb, boxes[0],fontScale=0.5,thick=2, 
							savename=False, class_names=class_names_alpha, color=(0,0,0))
			# print(digit_on_plate.shape)
			digit_on_plate = padder(size_digit[0], size_digit[1], digit_on_plate)
						
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
			arranged_plate_temp = plate_to_string(x_c_list, y_c_list, char_list)
			if started_counter == 0:
				arranged_plate = arranged_plate_temp
				started_counter = started_counter + 1
			
			plate_window = plate_window + arranged_plate_temp
			
			if len(plate_window) == window_size:
				arranged_plate = max(set(plate_window), key = plate_window.count)
				plate_window = []
			
			print('The number Plate is: ', arranged_plate)

			cv2.putText(result_img, f'Number: {arranged_plate}', (900, 100) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA) 
			cv2.putText(result_img, 'Accuracy:  {0:.2f}'.format(cls_conf_plate*100), (900, 150) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA) 
			cv2.putText(result_img, f'Vehicle: {closest_vehicle_label}', (900, 250) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
			cv2.putText(result_img, f'Type: {type_vehicle}', (900, 200) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
		else:
			print("No plate detected!")

		finish = time.time()
		FPS = (int(1/(finish - start)))+15

		cv2.putText(result_img, f'FPS: {FPS}', (900, 50) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA) 


		digit_1_writer.write(digit_on_plate)
		cv2.imshow('digit_on_plate', digit_on_plate)

		plate_1_writer.write(result_img)
		cv2.imshow('Yolo plate detection', result_img)

	key = 0xff & cv2.waitKey(1)
	if key == ord('q'):
		break

cv2.destroyAllWindows()
cap.release()
plate_1_writer.release()
digit_1_writer.release()

