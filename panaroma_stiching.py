import cv2 as cv
import numpy as np

img1 = cv.imread('./PanaromaImages/ImageSet1/image1.jpg')

img2 = cv.imread('./PanaromaImages/ImageSet1/image2.jpg',0)
print(img1.shape)
img1 = cv.resize(img1,(256,256))
img2 = cv.resize(img2,(256,256))



sift = cv.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)

sift = cv.xfeatures2d.SIFT_create()
kp2, des2 = sift.detectAndCompute(img2, None)



des1 = np.array(des1)
des2 = np.array(des2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

# flann = cv.FlannBasedMatcher(index_params,search_params)
bf = cv.BFMatcher()

matches = bf.knnMatch(des1,des2, k=2)
# matches = flann.knnMatch(des1,des2,k = 2)

matchesMask = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
	if m.distance < 0.7*n.distance:
		matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
					singlePointColor = (255,0,0),
					matchesMask = matchesMask,
					flags = 0)

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

cv.imshow("correspondences", img3)
cv.waitKey(0)