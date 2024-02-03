import cv2 as cv


def find_laser_plate_point(frame, center, debug=False):
    cropped = frame[
              round(center[1]) + 100:round(center[1]) + 200,
              round(center[0]) - 150:round(center[0]) + 150,
              :]

    cropped = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)

    mask1 = cv.inRange(cropped, (0, 55, 230), (20, 255, 255))
    mask2 = cv.inRange(cropped, (150, 55, 230), (180, 255, 255))
    cropped = mask1 | mask2

    if debug:
        cv.imshow('third point detection', cropped)

    _, laser = cv.threshold(cropped, 235, 255, cv.THRESH_BINARY)
    ret = cv.findNonZero(laser)

    cv.imshow("third point", cropped)
    if ret is None:
        raise Exception("third laser point not detected")
    laser_point = ret[0][0]

    return [round(laser_point[0] + center[0] - 150), round(laser_point[1] + center[1] + 100)]


def detect_laser_points(frame, ellipse, debug=False):
    blurred = cv.medianBlur(frame, 3)
    hsv = cv.cvtColor(blurred[
                      round(ellipse[0][1] - ellipse[1][0] / 2): round(ellipse[0][1] + ellipse[1][0] / 2),
                      round(ellipse[0][0] - ellipse[1][1] / 2): round(ellipse[0][0] + ellipse[1][1] / 2),
                      :], cv.COLOR_BGR2HSV)

    mask1 = cv.inRange(hsv, (0, 70, 230), (10, 255, 255))
    mask2 = cv.inRange(hsv, (160, 70, 230), (180, 255, 255))

    laser = mask1 | mask2

    intersections = []
    height, width = laser.shape

    for y in range(height):
        for x in range(width):
            if laser[y][x] > 0:
                intersections.append(
                    [round(x + ellipse[0][0] - ellipse[1][1] / 2), round(y + ellipse[0][1] - ellipse[1][0] / 2)])
                hsv[y][x] = [50, 255, 255]
                break

    if debug:
        cv.imshow("plate laser", hsv)
        cv.imshow("plate laser points", laser)

    return intersections
