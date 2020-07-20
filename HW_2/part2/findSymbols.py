import cv2
import imutils as imutils
import numpy as np
import matplotlib.pyplot as plt


def matchTemplate():

    templates = []
    names = []
    colors = [[255,204,204],[255,204,153],[255,255,102],[153,255,51],[0,255,0],
              [0,204,102],[0,153,153],[0,51,102],[0,0,51],[255, 51,51],[255, 128,0],
              [204,204,0],[76,153,0],[0,102,0],[0,51,25],[204,0,102]]

    for i in range(1, 17):
        if i < 10:
            templates.append(cv2.imread("symbols/00{}.png".format(i)))
            names.append("00{}".format(i))
        else:
            templates.append(cv2.imread("symbols/0{}.png".format(i)))
            names.append("0{}".format(i))

    image_filename = 'plan.png'
    img_bgr = cv2.imread(image_filename)
    # img_bgr.astype(np.uint8)
    template_filename = 'symbols/001.png'
    # tm_bgr = cv2.imread(template_filename)
    # tm_bgr = templates[15]

    for num, tm_bgr in enumerate(templates):
        color = colors[num]
        name = names[num]
        matching = cv2.matchTemplate(img_bgr, tm_bgr, cv2.TM_CCOEFF_NORMED)
        matching = cv2.normalize(matching, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # matching = cv2.resize(matching, (matching.shape[1] // 3, matching.shape[0] // 3))
        thresh = cv2.threshold(matching, np.max(matching) * 0.9, 255, cv2.THRESH_BINARY)[1]
        conts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_draw = img_bgr.copy()

        cv2.drawContours(img_draw, conts, -1, [0, 255, 0], 3)
        cv2.namedWindow('win', cv2.WINDOW_NORMAL)
        cv2.imshow('win', img_draw)
        cv2.waitKey(0)

        for cnt in conts:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_bgr, (x + w // 2, y + h // 2),
                          (x + w // 2 + tm_bgr.shape[1], y + h // 2 + tm_bgr.shape[0]),
                          color, 6)
            cv2.putText(img_bgr, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 2)
            # cv2.drawContours( img_bgr,conts,-1,[0,0,255],15)

        # img_bgr = cv2.resize(img_bgr, (img_bgr.shape[1] // 3, img_bgr.shape[0] // 3))

        # Showing images
        cv2.imshow('Window with example', img_bgr)
        cv2.waitKey(
            0)  # won't draw anything without this function. Argument - time in milliseconds, 0 - until key pressed
        cv2.imwrite('ex.png', img_bgr)


    # matching = cv2.matchTemplate(img_bgr, tm_bgr, cv2.TM_CCOEFF_NORMED)
    # matching = cv2.normalize(matching, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # # matching = cv2.resize(matching, (matching.shape[1] // 3, matching.shape[0] // 3))
    # thresh = cv2.threshold(matching, np.max(matching) * 0.9, 255, cv2.THRESH_BINARY)[1]
    # conts, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img_draw = img_bgr.copy()
    #
    # cv2.drawContours(img_draw, conts, -1, [0, 255, 0], 3)
    # cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    # cv2.imshow('win', img_draw)
    # cv2.waitKey(0)
    #
    # for cnt in conts:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     cv2.rectangle(img_bgr, (x + w // 2, y + h // 2), (x + w // 2 + tm_bgr.shape[1], y + h // 2 + tm_bgr.shape[0]),
    #                   [0, 0, 255], 6)
    #     # cv2.drawContours( img_bgr,conts,-1,[0,0,255],15)
    #
    # img_bgr = cv2.resize(img_bgr, (img_bgr.shape[1] // 3, img_bgr.shape[0] // 3))
    #
    # # Showing images
    # cv2.imshow('Window with example', img_bgr)
    # cv2.waitKey(0)  # won't draw anything without this function. Argument - time in milliseconds, 0 - until key pressed
    # cv2.imwrite('ex.png', img_bgr)


if __name__ == '__main__':
    matchTemplate()
