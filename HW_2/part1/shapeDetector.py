# import the necessary packages
import cv2


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
            color = [225, 225, 0]

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            if ar >= 0.95 and ar <= 1.05:
                shape = "square"
                color = [0, 255, 0]
            else:
                shape = "rectangle"
                color = [225, 204, 204]

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
            color = [0, 225, 225]

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
            color = [225, 0, 0]

        # return the name of the shape
        return shape, color
