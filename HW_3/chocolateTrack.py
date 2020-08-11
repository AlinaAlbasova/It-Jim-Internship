import cv2
import numpy as np


def chocoTrack():
    img = cv2.imread("marker.jpg", cv2.IMREAD_GRAYSCALE)
    # img = iu.resizeWithRescale(img, 60)

    cap = cv2.VideoCapture("find_chocolate.mp4")
    ret, frm = cap.read()

    # features
    orb = cv2.ORB_create()
    img_keypoints, img_descriptors = orb.detectAndCompute(img, None)
    # img = cv2.drawKeypoints(img, img_keypoints, img)

    # feature matching
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Setting video format. Google for "fourcc"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    # Setting up new video writer
    frames_per_second = 30
    # frm_rescaled = iu.resizeWithRescale(img, 60)
    image_size = (frm.shape[1], frm.shape[0])
    writer = cv2.VideoWriter('output_video.avi', fourcc, frames_per_second, image_size)

    while ret:
        # _, frame = cap.read()
        # frm = iu.resizeWithRescale(frm, 60)
        grey_frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)



        frm_keypoints, frm_descriptors = orb.detectAndCompute(grey_frm, None)
        # grey_frm = cv2.drawKeypoints(grey_frm, frm_keypoints, grey_frm)

        matches = bf.match(img_descriptors, frm_descriptors)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv2.drawMatches(img, img_keypoints, grey_frm, frm_keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # homography
        query_pts = np.float32([img_keypoints[m.queryIdx].pt for m in matches[:10]]).reshape(-1,1,2)
        train_pts = np.float32([frm_keypoints[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # perspective transform
        h,w = img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, matrix)

        homography = cv2.polylines(frm, [np.int32(dst)], True, (255,0,0), 3)

        writer.write(homography)
        cv2.imshow("Homography", homography)


        # cv2.imshow("Image", img)
        # cv2.imshow("Frame", grey_frm)

        # cv2.imshow("Img3", img3)

        key = cv2.waitKey(1)
        ret, frm = cap.read()
        if key == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    chocoTrack()
