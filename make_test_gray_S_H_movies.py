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


def get_gray_S_H(image):
    undistorted = undistort_image(image, cameraMatrix, distCoeffs)
    # test grayscale threshold method
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    gray_threshold = (180, 255)
    S_thresh = (90, 255)
    H_thresh = (15, 100)
    binary = np.zeros_like(gray)
    binary[(gray > gray_threshold[0]) & (gray <= gray_threshold[1]) \
           & (S > S_thresh[0]) & (S <= S_thresh[1])\
          & (H > H_thresh[0]) & (H <= H_thresh[1])] = 1
    return np.dstack((binary, binary, binary))*255


def get_gray_S(image):
    undistorted = undistort_image(image, cameraMatrix, distCoeffs)
    # test grayscale threshold method
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    gray_threshold = (180, 255)
    S_thresh = (90, 255)
    H_thresh = (15, 100)
    binary = np.zeros_like(gray)
    binary[(gray > gray_threshold[0]) & (gray <= gray_threshold[1]) \
           & (S > S_thresh[0]) & (S <= S_thresh[1])] = 1
    return np.dstack((binary, binary, binary))*255



gray_S_G = 'gray_S_H.mp4'
gray_S = 'gray_S.mp4'
clip1 = VideoFileClip("project_video.mp4")
gray_S_G_clip = clip1.fl_image(get_gray_S_H) #NOTE: this function expects color images!!
gray_S_G_clip.write_videofile(gray_S_G, audio=False)
gray_S_clip = clip1.fl_image(get_gray_S_H) #NOTE: this function expects color images!!
gray_S_clip.write_videofile(gray_S, audio=False)