import sys
sys.path.append('yolov3_detector')
from yolov3_custom_helper import yolo_detector, get_closest
from darknet import Darknet
sys.path.append('pytorch-YOLOv4')
from tool.darknet2pytorch import Darknet as DarknetYolov4
import argparse
import cv2,time,os
import numpy as np
import time
import torch

from tool.plateprocessing import find_coordinates, plate_to_string, padder, get_color
from tool.utils import alphanumeric_segemntor,plot_boxes_cv2
from tool.torch_utils import do_detect



def main(video_file, weight_v4_veh, weight_v4, weight_v4_alpha, use_web_int = False, use_cuda= True,window_size=5, frame_add_interval = 2):

	status  = None
	if use_web_int:
		from web_part import web_integration as webi
		# import web_integration as webi
		from web_part import notification as noti
		AuthID = '1544-1242-1878' 

	#################### Vehicle ####################
	cfg_v4_veh = 'pytorch-YOLOv4/cfg/yolov4.cfg'
	# weight_v4_veh = 'weights/yolov4.weights'

	m_vehicle = DarknetYolov4(cfg_v4_veh)
	m_vehicle.load_weights(weight_v4_veh)
	num_classes_veh = m_vehicle.num_classes

	class_names_veh = ['car','motorbike','bus','truck']
	print('Loading weights from %s... Done!' % (weight_v4_veh))

	#################### PLATE ####################

	cfg_v4 = 'pytorch-YOLOv4/cfg/yolo-obj.cfg'
	# weight_v4 = 'weights/yolo-obj_last.weights'

	m = DarknetYolov4(cfg_v4)
	m.load_weights(weight_v4)
	num_classes = m.num_classes
	class_names = ['plate']
	print('Loading weights from %s... Done!' % (weight_v4))

	#################### DIGIT ####################

	cfg_v4_alpha = 'pytorch-YOLOv4/cfg/digit.cfg'
	# weight_v4_alpha = 'weights/alphanumeric.weights'

	m_alpha = DarknetYolov4(cfg_v4_alpha)
	m_alpha.load_weights(weight_v4_alpha)
	num_classes_alpha = m_alpha.num_classes
	class_names_alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
	print('Loading weights from %s... Done!' % (weight_v4_alpha))

	if use_cuda:
		m_vehicle.cuda()
		m.cuda()
		m_alpha.cuda()

	############# READER/WRITER ##########	
	size = (1280,720)
	size_digit = (1200,1200)

	cap = cv2.VideoCapture(video_file)
	# plate_1_writer = cv2.VideoWriter('IvLabs_Day3.mp4',  cv2.VideoWriter_fourcc(*'mp4v'), 20, size) 
	# digit_1_writer = cv2.VideoWriter('digit_2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, size_digit) 
	# cap.set(3, 2048)
	# cap.set(4, 1536)

	########### PUT TEXT ###########
	fontScale = 1
	color = (0, 0, 0)  
	thickness = 2

	########## MAJORITY AND INCREASE FPS ##########
	plate_window = []
	type_window = []
	area_window = []
	vehicle_window = []

	 # Only the second frame will be read
	window_counter = 0
	frame_counter = 0
	started_counter = 0
	############################

	print("Starting Detection...")
	result_img = np.zeros((size[0], size[1], 3), dtype = np.uint8)
	arranged_plate = 'N/A'
	digit_on_plate = np.zeros_like(result_img)

	states_names = ['AN','AP','AR','AS','BR','CG','CH','DD','DL','GA','GJ','HP','HR','JH','JK','KA','KL','LA','LD','MH','ML','MN','MP','MZ','NL','OD','PB','PY','RJ','SK','TN','TR','TS','UK','UP','WB']
	# ret = True
	cv2.namedWindow("Yolo plate detection", cv2.WINDOW_NORMAL)

	while True:	
		ret, img = cap.read()


		# ret = True
		# img = cv2.imread('sih_number_plate/OCR/sample/admhrhfdyv.jpg')
		frame_counter = frame_counter + 1
		if not ret:
			break
		h,w = img.shape[0], img.shape[1]
		print(h,w)
		# print(frame_counter)
		if frame_counter % frame_add_interval == 0:
			frame_counter = 0
			sized = cv2.resize(img, (m.width, m.height))
			sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

			start = time.time()

			confidence_vehicle = 0.2
			boxes = do_detect(m_vehicle, sized, confidence_vehicle, 0.6, use_cuda)
			result_veh, conf_veh, coord_veh, labels_veh = plot_boxes_cv2(img, boxes[0],classes_to_detect=class_names_veh,fontScale=0.5,thick=2,savename=False)
			conf_veh = float(conf_veh)

			coordinates, closest_vehicle_label = get_closest(coord_veh, labels_veh)

			############# Plate ############


			boxes = do_detect(m, sized, 0.2, 0.6, use_cuda)
			result_img, cls_conf_plate, coord_plate, labels_plate = plot_boxes_cv2(result_veh, boxes[0],classes_to_detect=class_names,fontScale=0.5,thick=2, savename=False, class_names=class_names)
			cls_conf_plate = float(cls_conf_plate)

			coord_plate, closest_plate_label = get_closest(coord_plate, labels_plate)
			# digit_on_plate = np.zeros((size_digit[0], size_digit[1], 3), dtype = np.uint8)
			digit_on_plate = np.zeros((100, 100, 3), dtype = np.uint8)
			# cv2.rectangle(result_img, (int(0.09*h), 0),(int(0.4*h), 300),(255,255,255), thickness = -1)

			if coord_plate is not None:
				# x1, y1, x2, y2 = find_coordinates(img, coord_plate)
				x1,y1,x2,y2 = coord_plate[0],coord_plate[1],coord_plate[2],coord_plate[3]
				# print(x1,y1,x2,y2)
				plate_bb = img[y1:y2,x1:x2]
				# print(plate_bb.shape)
				area_box = abs((y1 - y2) * (x1 - x2))
				#print(plate_bb.shape)
				type_vehicle_temp = get_color(plate_bb)

				######### DETECT Digits ############
				
				sized = cv2.resize(plate_bb, (m_alpha.width, m_alpha.height))
				sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
				confidence = 0.6
				boxes = do_detect(m_alpha, sized, confidence , 0.6, use_cuda)
			
				digit_on_plate, _,_,_ = plot_boxes_cv2(plate_bb, boxes[0],classes_to_detect=class_names_alpha,fontScale=0.5,thick=2, savename=False, class_names=class_names_alpha, color=(0,0,0))
			
				alphanumerics,x_c_list,y_c_list = alphanumeric_segemntor(plate_bb, boxes[0],class_names=class_names_alpha)

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

				################ First Letter can never be a digit #######################
				if arranged_plate_temp[0] in ['0','1','2','3','4','5','6','7','8','9']:
					arranged_plate_temp = arranged_plate_temp[1:]
				# print('The number Plate is: ', arranged_plate)

				if started_counter == 0:
					# arranged_plate = arranged_plate_temp
					type_vehicle = type_vehicle_temp
					started_counter = started_counter + 1
					
				plate_window = plate_window + [arranged_plate_temp]
				type_window = type_window + [type_vehicle_temp]
				area_window = area_window + [area_box]
				vehicle_window = vehicle_window + [closest_vehicle_label]

				if frame_counter%window_size == 0:
					cv2.putText(result_img, f'Number: {arranged_plate_temp}', (int(0.1*h), 100) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)

				if len(plate_window) == window_size:
					if area_window[0] < area_window[-1]:
						status = 'entry'
						print("ENTERING")
					else:
						status = 'exit'
						print("LEAVING")

					if closest_vehicle_label != max(set(vehicle_window), key = vehicle_window.count):
						closest_vehicle_label = max(set(vehicle_window), key = vehicle_window.count)

					print(arranged_plate, max(set(plate_window)))
					if arranged_plate != max(set(plate_window), key = plate_window.count):
						arranged_plate = max(set(plate_window), key = plate_window.count)
					
						# print(arranged_plate[:2])
						cv2.putText(result_img, f'Number: {arranged_plate}', (int(0.1*h), 100) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)

						if arranged_plate[:2] in states_names:

							if use_web_int == True:	
								print("web_integration")
								cv2.imwrite('images/send_to_cloud.png', img)
								############### Check if car is registered ################
								registered, visits, block = webi.pull_data(AuthID, arranged_plate)
								if registered == True:
									reg = 1	 #For registered
									webi.collection_push_data(AuthID=AuthID,reg_number=arranged_plate,gate='gate1', view=status, time_date=webi.get_time())
								else:
									reg =0	# for non-registered
								if block == True:
									reg = 2	#Blacklisted
									noti.login(closest_vehicle_label,arranged_plate,1)
								########### Record Vehicle Data in Database ###############
								webi.push_data(gate='gate1', view=status, AuthID=AuthID, reg_number=arranged_plate, if_reg =reg, time_date=webi.get_time(), veh_type=closest_vehicle_label, visits=visits)
								print("--------------------------------------- SEND TO WEB: ", arranged_plate, "---------------------------------------")

					if type_vehicle != max(set(type_window), key = type_window.count):
						type_vehicle = max(set(type_window), key = type_window.count)

					plate_window = []
					type_window = []
					area_window = []
					vehicle_window = [] 
				cv2.putText(result_img, 'Accuracy:  {0:.2f}'.format(cls_conf_plate*100), (int(0.1*h), 150) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA) 
				cv2.putText(result_img, f'Vehicle: {closest_vehicle_label}', (int(0.1*h), 250) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
				cv2.putText(result_img, f'Type: {type_vehicle}', (int(0.1*h), 200) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)
				print("Plate", arranged_plate)
			else:
				print("No plate detected!")

			finish = time.time()
			FPS = (int((1.8*frame_add_interval)/(finish - start)))

			cv2.putText(result_img, f'FPS: {FPS}', (int(0.1*h), 50) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA) 
			
		# digit_1_writer.write(digit_on_plate)
		cv2.imshow('digit_on_plate', digit_on_plate)	
		# cv2.putText(result_img, f'Number: {arranged_plate}', (900, 100) , cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA)


		# plate_1_writer.write(result_img)
		# print(result_img.shape)
		cv2.imshow('Yolo plate detection', result_img)

		key = 0xff & cv2.waitKey(1)
		if key == ord('q'):
			break
	# plate_1_writer.release()
	cv2.destroyAllWindows()
	cap.release()
	


if __name__ == '__main__':

	video_file = 'videos/3.mp4'
	assert os.path.exists(video_file) , "Error with Video File"

	weight_v4_veh = 'weights/yolov4.weights'
	assert os.path.exists(weight_v4_veh) , "Error with vehicle weights"

	weight_v4_plate = 'weights/plate.weights'
	assert os.path.exists(weight_v4_plate) , "Error with License Plate weights"

	weight_v4_alpha = 'weights/alphanumeric.weights'
	assert os.path.exists(weight_v4_alpha) , "Error with OCR weights"

	use_web_int = True		# Use web integration 
	use_cuda= torch.cuda.is_available()		# use gpu if available
	window_size=5 # Mode of the list will be taken for these many samples

	main(video_file, weight_v4_veh, weight_v4_plate, weight_v4_alpha, use_web_int = use_web_int, use_cuda= use_cuda, window_size = window_size)

