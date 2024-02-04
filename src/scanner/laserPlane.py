import cv2 as cv


def find_laser_plate_point(frame, center, debug=False):
    """
    Find a laser point on the plate image
    :param frame: the frame
    :param center: the plate center on the image
    :param debug: debug mode flag
    :return: the coordinates of the point
    """
    # crop frame below ellipse center where laser line is flat
    cropped = frame[
              round(center[1]) + 100:round(center[1]) + 200,
              round(center[0]) - 150:round(center[0]) + 150,
              :]

    cropped = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)  # convert to hsv

    if debug:
        cv.imshow('third point detection', cropped)

    # laser value ranges to threshold the image
    mask1 = cv.inRange(cropped, (0, 55, 230), (20, 255, 255))
    mask2 = cv.inRange(cropped, (150, 55, 230), (180, 255, 255))
    laser = mask1 | mask2

    ret = cv.findNonZero(laser)  # simpy pick the first non-zero value

    if debug:
        cv.imshow("third point", laser)

    if ret is None:
        raise Exception("third laser point not detected")
    laser_point = ret[0][0]

    # extract point and convert to non-cropped coordinates
    return [round(laser_point[0] + center[0] - 150), round(laser_point[1] + center[1] + 100)]


def detect_laser_points(frame, ellipse, debug=False):
    """
    Detect all the laser points inside the ellipse on the frame
    :param frame:
    :param ellipse: the ellipse to contour the laser point search
    :param debug: debug mode flag
    :return: the list of laser points on the image
    """
    blurred = cv.medianBlur(frame, 3)  # blur image with 3x3 kernel
    # crop and convert to hsv
    hsv = cv.cvtColor(blurred[
                      round(ellipse[0][1] - ellipse[1][0] / 2): round(ellipse[0][1] + ellipse[1][0] / 2),
                      round(ellipse[0][0] - ellipse[1][1] / 2): round(ellipse[0][0] + ellipse[1][1] / 2),
                      :], cv.COLOR_BGR2HSV)

    # red laser mask ranges
    mask1 = cv.inRange(hsv, (0, 70, 225), (20, 255, 255))
    mask2 = cv.inRange(hsv, (155, 70, 225), (180, 255, 255))

    laser = mask1 | mask2

    laser_points = []
    height, width = laser.shape

    for y in range(height):
        for x in range(width):
            if laser[y][x] > 0:
                # pick leftmost laser point for each line and add it to the list
                laser_points.append(
                    [round(x + ellipse[0][0] - ellipse[1][1] / 2), round(y + ellipse[0][1] - ellipse[1][0] / 2)])
                if debug:
                    hsv[y][x] = [50, 255, 255]
                break

    if debug:
        cv.imshow("plate laser", hsv)
        cv.imshow("plate laser points", laser)

    return laser_points
