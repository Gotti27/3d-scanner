import cv2 as cv


def find_laser_plate_point(frame, center):
    cropped = frame[
              round(center[1]) + 100:round(center[1]) + 200,
              round(center[0]) - 150:round(center[0]) + 150,
              :]

    cropped = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    # cropped = cv.medianBlur(cropped, 5)

    mask1 = cv.inRange(cropped, (0, 60, 230), (20, 255, 255))
    mask2 = cv.inRange(cropped, (150, 60, 230), (180, 255, 255))
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
    cv.ellipse(frame, ellipse, (0, 255, 0), thickness=5)
    redChannel = frame[
                 round(ellipse[0][1] - ellipse[1][0] / 2): round(ellipse[0][1] + ellipse[1][0] / 2),
                 round(ellipse[0][0] - ellipse[1][1] / 2): round(ellipse[0][0] + ellipse[1][1] / 2),
                 2]
    redChannel = cv.medianBlur(redChannel, 5)
    _, laser = cv.threshold(redChannel, 235, 255, cv.THRESH_BINARY)
    laser = cv.morphologyEx(laser, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # laser = cv.morphologyEx(laser, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    intersections = []
    for i in range(0, len(laser)):
        for j in range(0, len(laser[i])):
            if laser[i][j] > 0:
                intersections.append(
                    [j + round(ellipse[0][0] - ellipse[1][1] / 2), i + round(ellipse[0][1] - ellipse[1][0] / 2)])
                break

    frame = laser
    cv.imshow("intersections", frame)
    return intersections
