import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
import cv2
from PIL import Image
import time

from helper import *
from models.enet.model import *
from models.emnist.model import *

from platedetect import img2str,image_crop_pad_resize



video_test_path = '/home/rohit/Downloads/1.mp4' #input('Enter path to video: ')
CUDA = torch.cuda.is_available()
alpha = 0.3
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
idx = ['0','1','2','3','4','5','6','7','8','9',
       'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
       'a','b','d','e','f','g','h','n','q','r','t']


transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean = mean, std = std)
			])
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,), (1,))])

cv2.namedWindow('Original',cv2.WINDOW_NORMAL)
cv2.namedWindow('plate',cv2.WINDOW_NORMAL)
cv2.namedWindow('segmented',cv2.WINDOW_NORMAL)


if CUDA:
	torch.cuda.set_device(0)
	print('Running on: ', torch.cuda.get_device_name(0))
else:
	print("Running on: CPU")



net = ENet(num_classes = 1)
net.load_state_dict(torch.load('saved_models/final_epoch9.pt', map_location = 'cpu'))

emnist_model = Net()
emnist_model.load_state_dict(torch.load("char_recognizer.pt"))



if CUDA:
	net.cuda()
	emnist_model.cuda()
net.eval()


cap = cv2.VideoCapture(video_test_path)

i = 0
resize_factor = 2
detected_texts = []

while(True) :
	inputtime = time.time()
	ret, frame = cap.read()
	_, frame_untouched = cap.read()
	#frame
	i+=1


	if i%1 == 0: 
		
		start_time = time.time()
		frame = cv2.resize(frame, tuple(np.flip(np.array(frame[:,:,1].shape))//resize_factor))
		frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_pil = Image.fromarray(frame2)
		frame_tf = transform(frame_pil)
		if CUDA:
			frame_tf = frame_tf.cuda()
		frame_tf = frame_tf.unsqueeze(0)
		out = net(frame_tf)
		out = torch.sigmoid(out)
		out = out.squeeze(1).detach().cpu()
		out = out.numpy()


		
		temp = np.zeros(frame.shape, np.uint8)
		temp[:, :, 1] = out[0] * 255
		
		segmented = cv2.addWeighted(frame, alpha, temp, (1 - alpha), 0.0)
		#cv2.imshow('segmentation_out', segmented)

		# cv2.imwrite(os.path.join(savePath, 'greenify_{}.png'.format(i)), segmented)
		

		#print(frame.shape, temp.shape)
		original, thresh, alphanumerics, dirty_plate_no_contour = img2str(frame_untouched, resize_factor,temp)
		#print('Detected Text: ', ocr_detection)
		#if ocr_detection is not '' and ocr_detection not in detected_texts:
		#	detected_texts.append(ocr_detection)
		# print("Total Boxes: ",len(alphanumerics))

		################################# EMNIST #################################
		b1 = alphanumerics[1]
		cropped = image_crop_pad_resize(dirty_plate_no_contour, b1[0],b1[1],pad =30 )
		cv2.imshow('cropped',cropped)
		ip = trans(cropped)

		ip = ip.reshape(1,1,128,128)
		if CUDA:
			ip = ip.cuda()	
		outputs_test = emnist_model(ip)
		_, pred = torch.max(outputs_test.data, 1)
		predx = pred.item()
		print("Recognized:", idx[predx])

		############################### EMNIST END ###############################


		#cv2.imshow('original',original)
		# original = cv2.resize(original,(segmented.shape[1],segmented.shape[0]))
		#cv2.imshow('thresh', thresh)
		# origseg = np.hstack((original,segmented))
		#cv2.imshow('output',origseg)
		#thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
		# thresh = cv2.resize(thresh,((thresh.shape[1]*3),thresh.shape[0]*3))
		# canvas = np.zeros((original.shape[0],original.shape[1]),dtype = 'uint8')
		# canvas[0:thresh.shape[0],0:thresh.shape[1]] = thresh
		# thresh = cv2.cvtColor(canvas,cv2.COLOR_GRAY2BGR)
		# final = np.hstack((origseg,thresh))

		# if len(detected_texts)>0:
		# 	final = cv2.putText(final, 'Detected: {}'.format(detected_texts[0]), (final.shape[1]*2//3,final.shape[0]*2//3), 
		# 		cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 255), 
		# 		2, cv2.LINE_AA)

		end_time = time.time()
		fps = round(1 / (end_time - start_time))

		segmented = cv2.putText(segmented, 'FPS: {}'.format(fps), (segmented.shape[1]*2//3,segmented.shape[0]*6//7), 
				cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 255), 
				2, cv2.LINE_AA)

		cv2.imshow('segmented',segmented)
		cv2.imshow('plate',thresh)
		cv2.imshow('Original',original)
		# print(segmented.shape)
		# writer.write(final)
		# cv2.waitKey(0)
		#print(text)
		
		if cv2.waitKey(1) & 0xff == ord('q'):
			
			break

	
	# else :
	# 	pass
# writer.release()		
cap.release()
cv2.destroyAllWindows()
