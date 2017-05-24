from matplotlib import image as mpimg
import cv2
import numpy as np
import random
from os import listdir
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# gets images used for calibration from file
def get_calibration_images():
    images = []
    for image_file in listdir('camera_cal'):
        image = mpimg.imread('camera_cal/'+image_file)
        images.append(image)
    return np.array(images)


calibration_images = get_calibration_images()

chess_board_size = (9,6)
obj_points = []
img_points = []

objp = np.zeros((9*6, 3), np.float32)
objp[:,:2] = np.mgrid[:9,:6].T.reshape(-1, 2)

def get_chess_corners(img, chess_board_size=(9,6), draw_result = False):
    nx, ny = chess_board_size
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret == True:
        # Draw and display the corners
        if draw_result:
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        return corners
    else:
        return None


# filter and process the chessboard images
proper_calibraiton_indexes = []
for i in range(len(calibration_images)):
    corners = get_chess_corners(calibration_images[i], draw_result=False)
    if type(corners) is not type(np.array([])):
        print("No proper chessboard was found in image", i)
    else:
        print("Image", i, "has a proper chessboard")
        proper_calibraiton_indexes.append(i)
        obj_points.append(objp)
        img_points.append(corners)


calibration_images = calibration_images[proper_calibraiton_indexes]


# Get the camera matrix and distortion coefficient from the image mappings we detected.
retval, cameraMatrix, distCoeffs, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, img_points, calibration_images[0].shape[0:2], None, None)

# a function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def undistort_image(img, cameraMatrix, distCoeffs):
    return cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)


test_images = [mpimg.imread('test_images/'+image_file) for image_file in listdir('test_images/')]


def gradient_and_color_decomposition(img, sobel_kernel=3, dir_thresh=(0.7, 1.3), mag_thresh=(50, 100)):
    undistorted = undistort_image(img, cameraMatrix, distCoeffs)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    
    
    gray_threshold = (180, 255)
    S_thresh = (90, 255)
    

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # for sobel gradient magnitude
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobelx_abs= np.absolute(sobelx)
    sobely_abs= np.absolute(sobely)
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    
    # for sobel gradient direction
    sobel_drctn = np.arctan2(sobely_abs, sobelx_abs)
    
    # merging the direciton and magnitude
    gradient_and_color = np.zeros_like(sobel_drctn)
    gradient_and_color[((scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1]) & \
        (sobel_drctn >= dir_thresh[0]) & (sobel_drctn <= dir_thresh[1])) | \
            ((gray > gray_threshold[0]) & (gray <= gray_threshold[1]) \
           & (S > S_thresh[0]) & (S <= S_thresh[1]))] = 1

    return np.dstack((gradient_and_color, gradient_and_color, gradient_and_color))*255
    
gradient_and_color_decomposition(test_images[0])


gray_S_G = 'gray_S_gradient.mp4'
clip1 = VideoFileClip("project_video.mp4")
gray_S_gradient = clip1.fl_image(gradient_and_color_decomposition) #NOTE: this function expects color images!!
gray_S_gradient.write_videofile(gray_S_G, audio=False)