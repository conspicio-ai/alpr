import cv2
import numpy as np

def find_boxes(thresh, drawplates, areathresh):
	total, labels, boxes, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
	if total > 1:
		if drawplates:
			thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
		cc = np.empty((0,4))
		i = 0
		while(i < total):
			x1 = boxes[i][0]
			y1 = boxes[i][1]
			x2 = x1 + boxes[i][2]
			y2 = y1 + boxes[i][3]
			if boxes[i][4] < areathresh:
				cc = np.append(cc, np.array([[x1,y1,x2,y2]]), axis = 0)
				if drawplates:
					cv2.rectangle(thresh, (x1, y1), (x2, y2), (0,0,255), 1)	
			i = i + 1
		return thresh, cc
	else:
		return thresh, np.empty((0,4))

def find_coordinates(img, boxes):
    width = img.shape[1]
    height = img.shape[0]
    
    for i in range(len(boxes)):
        box = boxes[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)
    return x1, y1, x2, y2


def order_points(pts):

	rect = np.zeros((4, 2), dtype = "float32")

	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):

	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped



def plate_detect(frame, boxes, drawplates, areathresh):
	rf = 1
	kernel = np.ones((5,5),np.uint8)

	x1, y1, x2, y2 = find_coordinates(frame, boxes)
	if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
		x2, y2 = 1, 1
	
	img = frame[y1:y2,x1:x2]
	alphanumerics = []
	Iclear = np.zeros((10,10))
	Iopen  = np.zeros((10,10))
	imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
	sh = imggray.shape

	#original = cv2.resize(original,(sh[1],sh[0]))
	#Image enhancement using morphological transformation
	ret,thresh = cv2.threshold(imggray,60,255,0)
	thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)

	#Contour detection for detecting the number plate area
	if (int(cv2.__version__[0]) < 4):
		x, contours, hierarchy = cv2.findContours(thresh, 1, 2)
	else:
		contours, hierarchy = cv2.findContours(thresh, 1, 2)
		
	if len(contours) >0:

		c = max(contours, key=cv2.contourArea)
		extLeft = tuple(c[c[:, :, 0].argmin()][0])
		extRight = tuple(c[c[:, :, 0].argmax()][0])
		extTop = tuple(c[c[:, :, 1].argmin()][0])
		extBot = tuple(c[c[:, :, 1].argmax()][0])
		rect = cv2.minAreaRect(c)
		pts11 = cv2.boxPoints(rect)
		box = np.int0(pts11)*rf
		p1,p2,p3,p4 = box

		pts = np.array([p1, p2, p3, p4], dtype = "float32")


		plate = four_point_transform(img,pts)
		
		imgg = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(imgg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		thresh = cv2.medianBlur(thresh, 3)
		
		thresh, digitbox = find_boxes(thresh, drawplates, areathresh)
	return thresh, digitbox
	
#cap = cv2.VideoCapture('/home/arihant/Downloads/1.mp4')

#locfile = open('/home/arihant/sih_number_plate-master1/locations.txt','r')

#coor = locfile.readline()

