from custom_helper import *


OUTPUT_SIZE = (1280,720)

############Classes to detect
CLASSES_TO_DETECT = ['bicycle', 'car', 'motorbike', 'truck']


	

if __name__ == '__main__':

	CUDA = torch.cuda.is_available()
	
	cfgfile = "cfg/yolov3.cfg"
	weightsfile = "yolov3.weights"


	model = Darknet(cfgfile)
	model.load_weights(weightsfile)
	model.net_info["height"] = 160
	inp_dim = int(model.net_info["height"])

	if CUDA:
		model.cuda()
	model.eval()


	
	frame = cv2.imread("test.jpg")#Give the frame here
	frame = cv2.resize(frame, OUTPUT_SIZE, interpolation = cv2.INTER_AREA)

	img, coordinates = yolo_output(frame.copy(),model, CLASSES_TO_DETECT, CUDA, inp_dim)
	
	closest_vehicle_coordinates = get_closest(coordinates)
	print(closest_vehicle_coordinates)

	cv2.imshow('yolo', img)
	cv2.imshow('original image', frame)

	cv2.waitKey(0)

	cv2.destroyAllWindows()