import cv2
import numpy as np

def detect(frm):
    frm = cv2.cvtColor(frm, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(frm)

    s = s + 100

    v = v + 70

    frm_new = cv2.merge((h, s, v))
    frm_new = cv2.cvtColor(frm_new, cv2.COLOR_HSV2BGR)
    frm = cv2.cvtColor(frm_new, cv2.COLOR_BGR2GRAY)

    ret, frm = cv2.threshold(frm, 90, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frm = cv2.morphologyEx(frm, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), dtype=np.uint8))
    frm = cv2.GaussianBlur(frm, (7, 7), 0)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(frame)
    #
    # h = h - 50
    #
    # s = s + 50
    # v = v - 10
    # frame= cv2.merge((h, s, v))
    # frame= cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    # frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -3)
    # #
    # frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), dtype=np.uint8))
    # #
    # frame = cv2.GaussianBlur(frame, (7, 7), 0)
    return frm


def run_shapes_detection():
    '''Simple function for cropping of a video and saving into a new file'''
    video_name = 'input_video.avi'
    cap = cv2.VideoCapture(video_name)
    ret,frm = cap.read()
    frm_count = 0

    # Setting video format. Google for "fourcc"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Setting up new video writer
    frames_per_second = 30
    image_size = (frm.shape[1],frm.shape[0])
    writer = cv2.VideoWriter('output_video.avi', fourcc, frames_per_second,image_size,0)

    # reading, writing and showing
    while ret:

        new_frm = detect(frm)

        writer.write(new_frm)
        cv2.imshow('Video frame',new_frm)
        cv2.waitKey(10)
        ret, frm = cap.read()
        frm_count+=1

    # don't forget to release the writer. Otherwise, the file may be corrupted.
    cap.release()
    writer.release()

    return 0

if __name__ == '__main__':
    run_shapes_detection()