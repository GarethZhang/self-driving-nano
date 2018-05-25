#importing some useful packages

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as n
import cv2
import os

opj = os.path.join

def calibrate_camera(cam_cali_dir, num_corners=(9,6)):
    '''
    Calibrate camera with chessboard images
    :param cam_cali_dir: directory that stores chessboard images
    :param num_corners: number of corners contained in the chessboard images; tuple
    :return: object and image points in 3D and 2D space
    '''

    # to store object and image objects
    objpoints = []
    imgpoints = []

    # object points if possible
    num_obj_w = num_corners[0]
    num_obj_h = num_corners[1]
    objp = n.zeros((num_obj_w*num_obj_h, 3), n.float32)
    objp[:,:2] = n.mgrid[0:num_obj_w,0:num_obj_h].T.reshape(-1,2) # x, y coordinates

    for fname in os.listdir(cam_cali_dir):
        img = mpimg.imread(opj(cam_cali_dir, fname))

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find corners on the chessboard
        ret, corners = cv2.findChessboardCorners(gray, (num_obj_w, num_obj_h), None)

        # if corners found, add object points and image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (num_obj_w, num_obj_h), corners, ret)
    return objpoints, imgpoints

def cal_undistort(img, objpoints, imgpoints):
    '''
    correction for image distortion
    :param img: distorted image
    :param objpoints: object points in 3D space
    :param imgpoints: image points in 2D space
    :return: an undistorted image
    '''
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = n.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = n.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = n.uint8(255 * abs_sobel / n.max(abs_sobel))
    grad_binary = n.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = n.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = n.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(n.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = n.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, n.pi / 2)):
    # Calculate gradient direction
    # Apply threshold
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = n.arctan2(n.absolute(sobely), n.absolute(sobelx))
    dir_binary = n.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = n.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def grad_threshold(img, ksize=11, do_plot=True):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(30, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(30, 255))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 255))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(n.pi / 6, n.pi / 3))
    s_binary = hls_select(img, thresh=(170, 255))

    # did NOT use gradient directions
    grad_img = s_binary | gradx

    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(img, cmap='gray')
        ax.set_title('before gradient thresholding')
        ax = fig.add_subplot(122)
        ax.imshow(grad_img, cmap='gray')
        ax.set_title('after gradient thresholding')
        plt.show()
    return grad_img


def crop_roi(img, vertices, do_plot=True):
    ignore_mask_color = 255
    mask = n.zeros_like(img)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked = cv2.bitwise_and(img, mask)
    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(masked, cmap='gray')
        ax.set_title('after masking ROI')
        plt.show()
    return masked


def masked_grad_image(images, do_plot=True):
    masked_grad_images = []
    for image in images:
        grad_image = grad_threshold(image, do_plot=do_plot)
        h, w = grad_image.shape[:2]
        h_offset = 50

        # get ROI
        lower_left = (100, h - h_offset)
        upper_left = (550, 420)
        upper_right = (666, 420)
        lower_right = (1200, h - h_offset)
        vertices = n.array([[lower_left, upper_left, upper_right, lower_right]], dtype=n.int32)

        # crop ROI
        masked_grad_image = crop_roi(grad_image, vertices, do_plot=do_plot)
        masked_grad_images.append(masked_grad_image)
    return masked_grad_images

def apply_perspective_transform(undist, src, dst, do_plot=True, cmap='gray', return_M=True):
    '''
    Apply perspective transform to 2D images
    :param undist: undistorted image
    :param src: source points
    :param dst: destination points
    :param do_plot: plot before and after change
    :param cmap: color map
    :param return_M: whether return M and Minv matrix
    :return: warped image with option of M and Minv matrix
    '''
    img_size = undist.shape[1], undist.shape[0]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    if do_plot:
        undist_cp = undist.copy()
        warped_cp = warped.copy()
        fig = plt.figure()
        ax = fig.add_subplot(211)
        cv2.polylines(undist_cp,n.int32([src]),True,(0,255,255),10)
        ax.imshow(undist_cp, cmap=cmap)
        ax.set_title('img before perspective transform')
        ax = fig.add_subplot(212)
        cv2.polylines(warped_cp,n.int32([dst]),True,(0,255,255),10)
        ax.imshow(warped_cp, cmap=cmap)
        ax.set_title('img after perspective transform')
    if return_M:
        return warped, M, Minv
    else:
        return warped


if __name__ == "__main__":
    # camera calibration
    cam_cali_dir = 'camera_cal/'
    objpoints, imgpoints = calibrate_camera(cam_cali_dir, num_corners=(9, 6))
    do_plot = False

    # pipeline starts
    for fname in os.listdir('test_images/'):
        # read in image
        image = mpimg.imread(opj('test_images/', fname))

        # correct for distortion
        undist_image = cal_undistort(image, objpoints=objpoints, imgpoints=imgpoints)

        # apply gradient and color threshold
        grad_image = grad_threshold(undist_image, do_plot=do_plot)
        h, w = grad_image.shape[:2]
        h_offset = 50

        # get ROI
        lower_left = (100, h - h_offset)
        upper_left = (550, 420)
        upper_right = (666, 420)
        lower_right = (1200, h - h_offset)
        vertices = n.array([[lower_left, upper_left, upper_right, lower_right]], dtype=n.int32)

        # crop ROI
        masked_grad_image = crop_roi(grad_image, vertices, do_plot=do_plot)

        # apply perspective transform
        src = n.array([[220, 720], [570, 470], [720, 470], [1100, 720]], dtype=n.float32)
        dst = n.array([[200, 720], [200, 0], [1080, 0], [1080, 720]], dtype=n.float32)
        binary_warped, M, Minv = apply_perspective_transform(masked_grad_image, src, dst, do_plot=do_plot)

        # find lines and fit second order polynomial
        histogram = n.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = n.dstack((binary_warped, binary_warped, binary_warped)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = n.int(histogram.shape[0] // 2)
        leftx_base = n.argmax(histogram[:midpoint])
        rightx_base = n.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = n.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = n.array(nonzero[0])
        nonzerox = n.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = n.int(n.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = n.int(n.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = n.concatenate(left_lane_inds)
        right_lane_inds = n.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = n.polyfit(lefty, leftx, 2)
        right_fit = n.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = n.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        offset = ((left_fitx[-1] + right_fitx[-1]) * 0.5 - midpoint) * xm_per_pix

        # Define conversions in x and y from pixels space to meters
        y_eval = n.max(ploty)
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = n.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = n.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / n.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / n.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        print(left_curverad, 'm', right_curverad, 'm')
        # Example values: 632.1 m    626.2 m

        # Create an image to draw the lines on
        warp_zero = n.zeros_like(binary_warped).astype(n.uint8)
        color_warp = n.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = n.array([n.transpose(n.vstack([left_fitx, ploty]))])
        pts_right = n.array([n.flipud(n.transpose(n.vstack([right_fitx, ploty])))])
        pts = n.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, n.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(undist_image, 1, newwarp, 0.3, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'Left lane curvature: ' + str(left_curverad) + 'm', (10, 50), font, 2, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(result, 'Right lane curvature: ' + str(right_curverad) + 'm', (10, 100), font, 2, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(result, 'Position: ' + str(offset) + 'm', (10, 150), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        plt.figure()
        plt.imshow(result)