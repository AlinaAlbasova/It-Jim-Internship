import cv2
import imutils as imutils
import numpy as np
import matplotlib.pyplot as plt


# step 1 - find blobs using code from previous homework.
# Here I will use example from homework review
# since my solution wasn't ideal.
from HW_2.part1.shapeDetector import ShapeDetector


def find_blobs(frm):
    # Making a copy of original frame and conversion to LAB
    processed_frm = frm.copy()
    processed_frm = cv2.cvtColor(processed_frm, cv2.COLOR_BGR2LAB)

    # Making a yellow mask
    yellow_mask = processed_frm[:, :, 2]
    yellow_mask = cv2.medianBlur(yellow_mask, 7)
    ret, yellow_mask = cv2.threshold(yellow_mask, 140, 255, cv2.THRESH_BINARY)  # good for non-equalized colors
    kernel = np.ones((17, 17), np.uint8)
    yellow_mask = cv2.erode(yellow_mask, kernel, iterations=1)

    # Making a green mask
    green_channel = 255 - processed_frm[:, :, 1]
    yellow_channel = processed_frm[:, :, 2]
    light_green_mask = np.uint8((1.6 * green_channel[:, :] + 0.4 * yellow_channel[:, :]) / 2)
    light_green_mask = cv2.medianBlur(light_green_mask, 7)
    ret, light_green_mask = cv2.threshold(light_green_mask, 135, 255,
                                          cv2.THRESH_BINARY)  # good for non-equalized colors
    kernel = np.ones((11, 11), np.uint8)
    light_green_mask = cv2.erode(light_green_mask, kernel, iterations=1)

    # Making another copy of original frame, equalization of each color channel and conversion to LAB
    processed_frm = frm.copy()
    for i in range(3):
        processed_frm[:, :, i] = cv2.equalizeHist(processed_frm[:, :, i])
    processed_frm = cv2.cvtColor(processed_frm, cv2.COLOR_BGR2LAB)

    # Making a green mask
    magenta_mask = processed_frm[:, :, 1]
    magenta_mask = cv2.medianBlur(magenta_mask, 11)
    ret, magenta_mask = cv2.threshold(magenta_mask, 185, 255, cv2.THRESH_BINARY)  # good for equalized colors

    # Making a copy of original frame and conversion to grayscale
    processed_frm = frm.copy()
    processed_frm = cv2.cvtColor(processed_frm, cv2.COLOR_BGR2GRAY)

    # Making a black mask
    black_mask = 255 - processed_frm
    ret, black_mask = cv2.threshold(black_mask, 230, 255, cv2.THRESH_BINARY)  # good for equalized colors
    kernel = np.ones((12, 12), np.uint8)
    black_mask = cv2.dilate(black_mask, kernel, iterations=1)

    # Masks binarization to apply Boolean OR operation
    yellow_mask = yellow_mask / 255
    light_green_mask = light_green_mask / 255
    magenta_mask = magenta_mask / 255
    black_mask = black_mask / 255

    # Making result binary mask
    result_mask = np.logical_or(yellow_mask, light_green_mask)
    result_mask = np.logical_or(result_mask, magenta_mask)
    result_mask = np.logical_or(result_mask, black_mask)

    # Making binary mask suitable for OpenCV showing and saving
    result_mask = np.uint8(result_mask * 255)

    # Making the grey scale image have three channels
    threshed = cv2.cvtColor(result_mask, cv2.COLOR_GRAY2BGR)

    return result_mask


def detect_shapes():
    video_name = 'input_video.avi'
    cap = cv2.VideoCapture(video_name)
    ret, frm = cap.read()
    frm_count = 0

    # Setting video format. Google for "fourcc"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Setting up new video writer
    frames_per_second = 30
    image_size = (frm.shape[1], frm.shape[0])
    writer = cv2.VideoWriter('output_video.avi', fourcc, frames_per_second, image_size)

    # reading, writing and showing
    while ret:
        mask = find_blobs(frm)
        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        sd = ShapeDetector()
        for c in cnts:
            c_flatten = np.array(c)
            c_flatten = c_flatten.flat
            if np.min(c_flatten) == 0:
                continue
            else:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                shape, color = sd.detect(c)
                cv2.drawContours(frm, [c], -1, color, 2)
                cv2.putText(frm, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)


        writer.write(frm)
        cv2.imshow('Video frame', frm)
        cv2.waitKey(10)
        ret, frm = cap.read()
        frm_count += 1

    # don't forget to release the writer. Otherwise, the file may be corrupted.
    cap.release()
    writer.release()

    return 0


if __name__ == '__main__':
    detect_shapes()
