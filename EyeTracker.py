#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy

# In[2]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[7]:


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    
    # for (x,y,w,h) in faces:
    #     cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        # Arguements => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness
        
        # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
      #roi_gray = gray[y:y+h, x:x+w]
      #roi_color = frame[y:y+h, x:x+w]
        # Detect eyes now
        #eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # Now draw rectangle over the eyes
    for (ex, ey, ew, eh) in eyes:
      cv2.rectangle(frame, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return frame    


# In[8]:


video_capture = cv2.VideoCapture(0)
# Run the infinite loop
while True:
  # Read each frame
  _, frame = video_capture.read()
  # Convert frame to grey because cascading only works with greyscale image
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Call the detect function with grey image and colored frame
  #blur = cv2.GaussianBlur(gray,(5,5),0)
  
  canvas = detect(gray, frame)
  #print(numpy.shape(canvas))
  # Show the image in the screen
  cv2.imshow("Video", canvas)
  # Put the condition which triggers the end of program
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




