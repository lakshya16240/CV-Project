import numpy as mp
import cv2 as cv
import math

face_cascade = cv.CascadeClassifier('/Users/lakshya/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv.CascadeClassifier('/Users/lakshya/opencv/data/haarcascades/haarcascade_eye.xml')

def detect(gray, frame):
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

		roi_gray = gray[y:int((y+h)*0.7), x:x+w]
		roi_color = frame[y:int((y+h)*0.7), x:x+w]

		eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)

		area = (x+w)*(y+h)
		print("Area",area)
		distance = 8414.7 * ((area**(1/2)) ** (-1))    # formula source - https://drive.google.com/file/d/0B-2k62V_M6QHWTU5R2N4QUJtZVE/view
		print("distance ",distance)
		
		for (ex, ey, ew, eh) in eyes:
			cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)

	return frame


video_capture = cv.VideoCapture(0)
# Run the infinite loop
while True:
	# Read each frame
	_, frame = video_capture.read()
	# Convert frame to grey because cascading only works with greyscale image
	gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	# Call the detect function with grey image and colored frame
	canvas = detect(gray, frame)
	# Show the image in the screen
	cv.imshow("Video", canvas)
	# Put the condition which triggers the end of program
	if cv.waitKey(1) & 0xFF == ord('q'):
		break
video_capture.release()
cv.destroyAllWindows()