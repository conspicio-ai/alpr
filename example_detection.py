import sys
sys.path.append('yolov3_detector')
from custom_helper import *


OUTPUT_SIZE = (1280,720)

############ Classes to detect
############ Classes can be found at coco.names files

CLASSES_TO_DETECT = ['bicycle', 'car', 'motorbike','truck']


if __name__ == '__main__':

	CUDA = torch.cuda.is_available()
	
	cfgfile = "yolov3_detector/cfg/yolov3.cfg"
	weightsfile = "yolov3_detector/yolov3.weights"
	names_file = "yolov3_detector/data/coco.names"

	model = Darknet(cfgfile)
	model.load_weights(weightsfile)
	model.net_info["height"] = 160
	inp_dim = int(model.net_info["height"])

	if CUDA:
		model.cuda()
	model.eval()


	
	frame = cv2.imread("/home/rohit/Pictures/vlcsnap-2020-03-28-18h06m43s854.png") #Give the frame here
	frame = cv2.resize(frame, OUTPUT_SIZE, interpolation = cv2.INTER_AREA)

	img, coordinates = yolo_output(frame.copy(),model, CLASSES_TO_DETECT, CUDA, inp_dim, names_file, confidence=0.21, nms_thesh=0.41)
	
	closest_vehicle_coord = get_closest(coordinates)
	h_yolo, w_yolo = closest_vehicle_coord[3] - closest_vehicle_coord[1], closest_vehicle_coord[2]-closest_vehicle_coord[0]
	print(closest_vehicle_coord)
	img = frame[closest_vehicle_coord[1]:closest_vehicle_coord[1]+h_yolo,closest_vehicle_coord[0]:closest_vehicle_coord[0]+w_yolo]
	cv2.imshow('yolo_largest', img)
	cv2.imshow('original image', frame)

	cv2.waitKey(0)

	cv2.destroyAllWindows()