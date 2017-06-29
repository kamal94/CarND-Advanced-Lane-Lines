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
        pass
    else:
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
def undistort_image(img, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs):
    return cv2.undistort(img, cameraMatrix, distCoeffs, None, cameraMatrix)

# Here we strip down the function to its simplest possible form
def find_lanes_binary(img, sobel_kernel=3, dir_thresh=(0.7, 1.3), mag_thresh=(50, 100)):
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

    return gradient_and_color


# Warping the image
src = np.array([[[529, 493]], [[759, 493]], [[235, 687]], [[1072, 687]]], dtype=np.float32)
dst = np.array([[[250, 450]], [[1080, 450]], [[250, 685]], [[1080, 685]]], dtype=np.float32)
warping_M = cv2.getPerspectiveTransform(src, dst)
unwarping_M = cv2.getPerspectiveTransform(dst, src)

# 
def warp_lines(img, M=warping_M):
    img_size = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

def warp_binary(img,):
    return warp_lines(find_lanes_binary(img))




test_images = [mpimg.imread('test_images/'+image_file) for image_file in listdir('test_images/')]

def get_lane_fit_image(image, plot=False):
    binary_warped = warp_binary(image)
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Choose the number of sliding windows
    nwindows = 15
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 30
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin


        # Draw the windows on the visualization image 
        cv2.rectangle(out_img,(win_xleft_low, win_y_low),(win_xleft_high, win_y_high), color=(0,255,0)) 
        cv2.rectangle(out_img,(win_xright_low, win_y_low),(win_xright_high, win_y_high), color=(0,255,0)) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
    
    binary = warp_binary(undistort_image(image, cameraMatrix, distCoeffs))
    binary_image = np.dstack((binary, binary, binary))*255
    
    
    undist = undistort_image(image)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, righty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the binary_warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, unwarping_M , (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    

    lefty_max = np.max(lefty)
    righty_max = np.max(rightx)
    y_eval = np.max([lefty_max, righty_max])
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
    / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
    / np.absolute(2*right_fit_cr[0])
    avg_curverad = (left_curverad + right_curverad)/2
    if avg_curverad > 3000.:
        avg_curverad = 0.
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,"curvature radius: " +str(avg_curverad),(500,700), font, 1,(255,255,255),2)

    mid_point_btwn_lanes = (rightx[0] + leftx[0])//2
    off_center = midpoint - mid_point_btwn_lanes
    off_center_in_meters = off_center * 3.7/700
    off_center_caption = "Car offcenter by " + str(np.abs(off_center_in_meters))[:5]
    if off_center_in_meters < 0:
        off_center_caption += "m R"
    else:
        off_center_caption += "m L"
        
    cv2.putText(result, off_center_caption,(500,650), font, 1,(255,255,255),2)
    cv2.line(result, (mid_point_btwn_lanes, result.shape[0]-100), \
             (midpoint, result.shape[0]-100), (255, 0, 0), thickness=5)
    cv2.line(result, (mid_point_btwn_lanes, result.shape[0]-80), \
             (mid_point_btwn_lanes, result.shape[0]-120), (0, 0, 255), thickness=2)

    # plot if required
    if plot:
        import matplotlib.pyplot as plt
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(25, 10))
        f.tight_layout()
        mark_size = 3
        ax1.imshow(image)
        ax2.set_title('Original', fontsize=30)
        ax2.imshow(out_img)
        ax2.set_title('Warped', fontsize=30)
        ax3.plot(leftx, lefty, 'o', color='red', markersize=mark_size)
        ax3.plot(rightx, righty, 'o', color='blue', markersize=mark_size)
        ax3.plot(left_fitx, lefty, color='green', linewidth=3)
        ax3.plot(right_fitx, righty, color='green', linewidth=3)
        ax3.invert_yaxis() # to visualize as we do the images
        ax3.set_title('Polynomial fit', fontsize=30)
        ax4.imshow(result)
        ax4.set_title('Result', fontsize=30)
    
    
    return result


final = 'final.mp4'
clip1 = VideoFileClip("project_video.mp4")
final_movie = clip1.fl_image(get_lane_fit_image) #NOTE: this function expects color images!!
final_movie.write_videofile(final, audio=False)