import cv2 as cv


def find_laser_plate_point(frame, center):
    cropped = frame[
              round(center[1]) + 100:round(center[1]) + 200,
              round(center[0]) - 150:round(center[0]) + 150,
              :]

    cropped = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    # cropped = cv.medianBlur(cropped, 5)

    mask1 = cv.inRange(cropped, (0, 55, 230), (20, 255, 255))
    mask2 = cv.inRange(cropped, (150, 55, 230), (180, 255, 255))
    cropped = mask1 | mask2

    cv.imshow('test', cropped)

    _, laser = cv.threshold(cropped, 235, 255, cv.THRESH_BINARY)

    middle = cv.findNonZero(laser)[0][0]
    cv.drawMarker(frame, middle, (0, 255, 0), cv.MARKER_TILTED_CROSS, 15, 2)

    cv.imshow("third point", cropped)
    if middle is None:
        return None

    return [round(middle[0] + center[0] - 150), round(middle[1] + center[1] + 100)]


def detect_laser_points(frame, ellipse):
    test = cv.medianBlur(frame, 3)
    test = cv.cvtColor(test[
                       round(ellipse[0][1] - ellipse[1][0] / 2): round(ellipse[0][1] + ellipse[1][0] / 2),
                       round(ellipse[0][0] - ellipse[1][1] / 2): round(ellipse[0][0] + ellipse[1][1] / 2),
                       :], cv.COLOR_BGR2HSV)
    cv.ellipse(frame, ellipse, (0, 255, 0), thickness=5)
    redChannel = frame[
                 round(ellipse[0][1] - ellipse[1][0] / 2): round(ellipse[0][1] + ellipse[1][0] / 2),
                 round(ellipse[0][0] - ellipse[1][1] / 2): round(ellipse[0][0] + ellipse[1][1] / 2),
                 2]

    hej = test.copy()

    redChannel = cv.medianBlur(redChannel, 3)

    # _, laser = cv.threshold(redChannel, 255, 255, cv.THRESH_BINARY)
    # laser = cv.morphologyEx(laser, cv.MORPH_DILATE, np.ones((5, 5), np.uint8))
    # laser = cv.morphologyEx(laser, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

    mask1 = cv.inRange(test, (0, 70, 230), (10, 255, 255))
    mask2 = cv.inRange(test, (160, 70, 230), (180, 255, 255))

    test = mask1 | mask2
    # test = cv.morphologyEx(test, cv.MORPH_ERODE, np.ones((5, 5), np.uint8))

    intersections = []
    '''
    for i in range(0, len(test)):
        flag = False
        for j in range(0, len(test[i])):
            if test[i][j] > 0 and not flag:
                intersections.append(
                    [j + round(ellipse[0][0] - ellipse[1][1] / 2), i + round(ellipse[0][1] - ellipse[1][0] / 2)])
                flag = True
                break
            elif test[i][j] > 0:
                pass
                # laser[i][j] = 0
    '''

    height, width = test.shape

    for y in range(height):
        flag = False
        for x in range(width):
            if test[y][x] > 0 and not flag:
                intersections.append(
                    [x + round(ellipse[0][0] - ellipse[1][1] / 2), y + round(ellipse[0][1] - ellipse[1][0] / 2)])
                flag = True
                hej[y][x] = [50, 255, 255]
                break
            elif test[y][x] > 0:
                pass
                # laser[i][j] = 0

    # frame = laser
    # cv.imshow("intersections", laser)
    cv.imshow("plate laser", hej)
    cv.imshow("red", test)

    return intersections
