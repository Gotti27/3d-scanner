import cv2 as cv
import numpy as np


def find_rectangle(frame, contours2, w, h):
    _, test = cv.threshold(cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY), 90, 255, cv.THRESH_BINARY_INV)
    test = cv.morphologyEx(test, cv.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, hierarchy = cv.findContours(test, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    rects = []
    # cv.drawContours(r, contours, -1, (0, 255, 0), 5)

    for i, contour in enumerate(contours):

        # Approximate the contour to a polygon
        polygon = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)

        # cv.polylines(r, polygon, True, (0, 255, 0), 5)

        '''
        and abs(
                1 - cv.contourArea(polygon) / (cv.boundingRect(polygon)[2] * cv.boundingRect(polygon)[3])) < 0.14
        '''
        # cv.fillPoly(r, contour, (0, 255, 0))

        area = cv.contourArea(contour)

        if len(polygon) == 4 and area > 100000 and any([p[0][1] < 200 for p in polygon]):
            # polygon = [p[0] for p in polygon]
            a, b, c, d = polygon
            # cv.drawMarker(r, a[0], (255, 255, 0), cv.MARKER_CROSS, 30, 5)
            # cv.drawMarker(r, b[0], (0, 255, 0), cv.MARKER_CROSS, 30, 5)
            # cv.drawMarker(r, c[0], (255, 0, 0), cv.MARKER_CROSS, 30, 5)
            # cv.drawMarker(r, d[0], (0, 0, 255), cv.MARKER_CROSS, 30, 5)

            # e = cv.fitEllipse(contour)
            # cv.ellipse(r, e, (0, 255, 0), 5)

            # cv.polylines(r, polygon, True, (0, 255, 0), 10)

            '''
            for p in polygon:
                print(p)
            '''
            # cv.drawMarker(r, p, (0, 0, 255), cv.MARKER_CROSS, 30, 3)

            rects.append(polygon)
            # cv.drawContours(r, [contour], -1, (0, 255, 0),
            #                5)

    rects.sort(key=lambda x: cv.contourArea(x))
    return rects[0]


def process_rectangle(rectangle, frame, original, mtx, dist):
    hej = frame.copy()
    a, b, c, d = rectangle

    obj_points = [[100, 100, 0], [230, 100, 0], [230, 330, 0], [100, 330, 0]]
    pln_points = [a[0], b[0], c[0], d[0]]

    cv.drawMarker(hej, a[0], (255, 255, 0), cv.MARKER_CROSS, 30, 5)
    cv.drawMarker(hej, b[0], (0, 255, 0), cv.MARKER_CROSS, 30, 5)
    cv.drawMarker(hej, c[0], (255, 0, 0), cv.MARKER_CROSS, 30, 5)
    cv.drawMarker(hej, d[0], (0, 0, 255), cv.MARKER_CROSS, 30, 5)

    #

    H = cv.findHomography(np.array(pln_points), np.array([[o[0], o[1]] for o in obj_points]))

    _, r, t = cv.solvePnP(np.array(obj_points, dtype=np.float32), np.array(pln_points, dtype=np.float32),
                          mtx, dist, flags=cv.SOLVEPNP_IPPE)

    transformed = cv.warpPerspective(frame.copy(), H[0], (1000, 1000))

    '''
    cv.line(transformed, [500, 500], [500 + 300, 500], (255, 0, 0), 5)
    cv.line(transformed, [500, 500], [500, 500 + 300], (0, 255, 0), 5)
    cv.line(transformed, [500, 500], [500 - 300, 500], (255, 255, 0), 5)
    cv.line(transformed, [500, 500], [500, 500 - 300], (0, 0, 255), 5)
    cv.drawMarker(transformed, [500, 500], (0, 0, 0), cv.MARKER_STAR, thickness=5)
    # obj_points = list(map(lambda i: list(i).append(0), obj_points))

    obj_points = np.array(obj_points, dtype=np.float32)
    pln_points = np.array(pln_points, dtype=np.float32)

    pln_points = np.append(pln_points, np.zeros((len(pln_points), 1)), axis=1)


    '''

    # cv.drawContours(r, [rects[0]], -1, (255, 0, 0), 5)

    # cv.drawContours(r, rects[3], -1, (0, 255, 0), 5)

    cv.imshow('hej', hej)
    cv.imshow('t', transformed)
