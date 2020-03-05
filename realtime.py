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

from platedetect import *
#cv2.namedWindow('segmentation_out',cv2.WINDOW_NORMAL)
cv2.namedWindow('segmented',cv2.WINDOW_NORMAL)

torch.cuda.set_device(0)
print(torch.cuda.get_device_name(0))

alpha = 0.3
# mean = [0.28689554, 0.32513303, 0.28389177]
# std = [0.18696375, 0.19017339, 0.18720214]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean = mean, std = std)
			])


net = ENet(num_classes = 1)
net.load_state_dict(torch.load('saved_models/final_epoch9.pt', map_location = 'cpu'))
net.cuda()
net.eval()

video_test_path = input('Enter path to video: ')
# video_test_path = '/home/himanshu/Downloads/2.mp4'
cap = cv2.VideoCapture(video_test_path)
# savePath =  "/home/himanshu/sih_number_plate/sem_seg_pytorch/thresh_imgs"

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# writer = cv2.VideoWriter('video_result/output_result_1.mp4', fourcc, 15, (540, 2880))

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
		# cv2.imwrite(os.path.join(savePath, 'orig_{}.png'.format(i)) , frame)

		frame = cv2.resize(frame, tuple(np.flip(np.array(frame[:,:,1].shape))//resize_factor))
		frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# frame2 = cv2.resize(frame2, (338,600))
		frame_pil = Image.fromarray(frame2)
		frame_tf = transform(frame_pil)
		frame_tf = frame_tf.cuda()
		frame_tf = frame_tf.unsqueeze(0)
		out = net(frame_tf)
		
		out = torch.sigmoid(out)
		out = out.squeeze(1).detach().cpu()
		# t = torch.Tensor([0.5])
		# out = (out > t)
		out = out.numpy()
		#end_time = time.time()


		
		temp = np.zeros(frame.shape, np.uint8)
		temp[:, :, 1] = out[0] * 255
		
		segmented = cv2.addWeighted(frame, alpha, temp, (1 - alpha), 0.0)
		#cv2.imshow('segmentation_out', segmented)

		# cv2.imwrite(os.path.join(savePath, 'greenify_{}.png'.format(i)), segmented)
		

		#print(frame.shape, temp.shape)
		original, thresh, ocr_detection = img2str(frame_untouched, resize_factor,temp)
		print('Detected Text: ', ocr_detection)
		if ocr_detection is not '' and ocr_detection not in detected_texts:
			detected_texts.append(ocr_detection)
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

		# end_time = time.time()
		# fps = round(1 / (end_time - start_time))+30

		# final = cv2.putText(final, 'FPS: {}'.format(fps), (final.shape[1]*2//3,final.shape[0]*6//7), 
		# 		cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 255), 
		# 		2, cv2.LINE_AA)

		cv2.imshow('segmented',segmented)
		cv2.imshow('plate',thresh)
		cv2.imshow('Original',original)
		# print(final.shape)
		# writer.write(final)
		#v2.waitKey(0)
		#print(text)
		
		if cv2.waitKey(1) & 0xff == ord('q'):
			
			break

	
	# else :
	# 	pass
# writer.release()		
cap.release()
cv2.destroyAllWindows()
