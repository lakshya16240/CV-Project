import cv2 as cv
import numpy as np


im1 =  cv.imread("/Users/lakshya/Desktop/im1.jpg");
print(im1.shape)

# im2 = np.array([[[0 for i in range(im1.shape[2])]for j in range(im1.shape[1])]for k in range(im1.shape[0])],dtype=np.uint8)
# for i in range(im1.shape[0]):
# 	for j in range(im1.shape[1]-1,2,-1):
# 			im2[i][j] = im1[i][j-3]



im2 =  cv.imread("/Users/lakshya/Desktop/im2.jpg");
 
# Convert images to grayscale
im1_gray = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
im2_gray = cv.cvtColor(im2,cv.COLOR_BGR2GRAY)
 
# Find size of image1
sz = im1.shape
 
# Define the motion model
warp_mode = cv.MOTION_EUCLIDEAN
 
# Define 2x3 or 3x3 matrices and initialize the matrix to identity
if warp_mode == cv.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)
 
# Specify the number of iterations.
number_of_iterations = 5000;
 
# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;
 
# Define termination criteria
criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
 
if warp_mode == cv.MOTION_HOMOGRAPHY :
    # Use warpPerspective for Homography 
    im2_aligned = cv.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)
else :
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP);
 
# Show final results
cv.imshow("Image 1", im1)
cv.imshow("Image 2", im2)
cv.imshow("Aligned Image 2", im2_aligned)
cv.waitKey(0)