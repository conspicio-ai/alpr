import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('yolov3_detector')
import os
import cv2
from PIL import Image
import time

from helper import *
from models.enet.model import *
from models.emnist.model import *

from custom_helper import *
from platedetect import img2str,image_crop_pad_resize


video_test_path = '/home/rohit/Videos/5.mp4' #input('Enter path to video: ')


def single_letter_ocr(image,CUDA):
	idx = ['0','1','2','3','4','5','6','7','8','9',
	   'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
	   'a','b','d','e','f','g','h','n','q','r','t']
	trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])
	ip = trans(image)
	ip = ip.reshape(1,1,128,128)
	if CUDA:
		ip = ip.cuda()	
	outputs_test = emnist_model(ip)
	_, pred = torch.max(outputs_test.data, 1)
	predx = pred.item()
	return idx[predx]

def yolo_detector(frame,CUDA,INPUT_SIZE = (1280,720)):
	
	CLASSES_TO_DETECT = ['bicycle', 'car', 'motorbike', 'truck', 'person', 'dog']
	
	frame = cv2.resize(frame, INPUT_SIZE, interpolation = cv2.INTER_AREA)

	img, coordinates,labels = yolo_output(frame.copy(),yolo_model, CLASSES_TO_DETECT, CUDA, 
		inp_dim, names_file, confidence=0.21, nms_thesh=0.41)
	
	closest_vehicle_coord, closest_vehicle_label = get_closest(coordinates, labels)
	h_yolo, w_yolo = closest_vehicle_coord[3] - closest_vehicle_coord[1], closest_vehicle_coord[2]-closest_vehicle_coord[0]
	img = frame[closest_vehicle_coord[1]:closest_vehicle_coord[1]+h_yolo,closest_vehicle_coord[0]:closest_vehicle_coord[0]+w_yolo]

	return img,closest_vehicle_label

## Specify for yolo
cfgfile = "yolov3_detector/cfg/yolov3.cfg"
weightsfile = "yolov3_detector/yolov3.weights"
names_file = "yolov3_detector/data/coco.names"

CUDA = torch.cuda.is_available()
alpha = 0.3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean = mean, std = std)
			])

cv2.namedWindow('original',cv2.WINDOW_NORMAL)
cv2.namedWindow('plate',cv2.WINDOW_NORMAL)
cv2.namedWindow('segmented',cv2.WINDOW_NORMAL)
cv2.namedWindow('yolo_dete',cv2.WINDOW_NORMAL)


if CUDA:
	torch.cuda.set_device(0)
	print('Running on: ', torch.cuda.get_device_name(0))
else:
	print("Running on: CPU")


yolo_model = Darknet(cfgfile)
yolo_model.load_weights(weightsfile)
yolo_model.net_info["height"] = 160
inp_dim = int(yolo_model.net_info["height"])

net = ENet(num_classes = 1)
net.load_state_dict(torch.load('saved_models/final_epoch9.pt', map_location = 'cpu'))

emnist_model = Net()
emnist_model.load_state_dict(torch.load("char_recognizer.pt")) # download this weights using instructions given in README.md

if CUDA:
	net.cuda()
	emnist_model.cuda()
	yolo_model.cuda()

net.eval()
emnist_model.eval()
yolo_model.eval()

cap = cv2.VideoCapture(video_test_path)

frame_skip_val = 0
resize_factor = 1
detected_texts = []

while(True) :
	inputtime = time.time()
	ret, frame = cap.read()
	_, frame_untouched = cap.read()
	frame_skip_val+=1


	if frame_skip_val%1 == 0: 
		
		start_time = time.time()
		# frame = cv2.resize(frame, tuple(np.flip(np.array(frame[:,:,1].shape))//resize_factor))
		frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# frame_pil = Image.fromarray(frame2)

		
	################################# YOLO #################################

		img_yolo, closest_vehicle_label = yolo_detector(frame2,CUDA,INPUT_SIZE = (1920,1080))
				
	################################# ENET #################################

		ENET_input_size = (300, 300)
		img_yolo_to_enet = cv2.resize(img_yolo, ENET_input_size)
		frame_tf = transform(img_yolo_to_enet)
		if CUDA:
			frame_tf = frame_tf.cuda()
		frame_tf = frame_tf.unsqueeze(0)
		out = net(frame_tf)
		out = torch.sigmoid(out)
		out = out.squeeze(1).detach().cpu()
		out = out.numpy()
		
	############################ Contour Extract ###########################

		temp = np.zeros(img_yolo_to_enet.shape, np.uint8)
		temp[:, :, 1] = out[0] * 255
		# print(temp.shape,frame_untouched.shape)
		temp = cv2.resize(temp, (img_yolo.shape[1],img_yolo.shape[0]))
		segmented = cv2.addWeighted(img_yolo, alpha, temp, (1 - alpha), 0.0)
		original, thresh, alphanumerics,dirty_plate_no_contour = img2str(img_yolo, resize_factor,temp)


	################################# EMNIST #################################

		if len(alphanumerics) == 10:
			detected_plate_info = []
			for cnt_box in alphanumerics:
				cropped = image_crop_pad_resize(dirty_plate_no_contour, cnt_box[0],cnt_box[1],pad =30 )
				detected_plate_info.append(single_letter_ocr(cropped,CUDA))

			detected_plate_info_string = '{}{} {}{} {}{} {}{}{}{}'.format(*detected_plate_info)

			position = 2
			cv2.imshow('',image_crop_pad_resize(dirty_plate_no_contour, alphanumerics[position][0],alphanumerics[position][1],pad =30))
			print(detected_plate_info_string, closest_vehicle_label)

	################################# FPS+IMSHOWS #################################

		end_time = time.time()
		fps = round(1 / (end_time - start_time))

		segmented = cv2.putText(segmented, 'FPS: {}'.format(fps), (segmented.shape[1]*2//3,segmented.shape[0]*6//7), 
				cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 255), 
				2, cv2.LINE_AA)

		cv2.imshow('yolo_dete',cv2.cvtColor(img_yolo, cv2.COLOR_RGB2BGR))	
		cv2.imshow('segmented',segmented)
		cv2.imshow('plate',thresh)
		# cv2.imshow('original',temp)
		
		if cv2.waitKey(1) & 0xff == ord('q'):			
			break
	
cap.release()
cv2.destroyAllWindows()
