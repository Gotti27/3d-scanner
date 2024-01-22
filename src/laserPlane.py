import cv2 as cv


def detect_laser(frame):
    redChannel = frame[:, :, 2].copy()
    _, laser = cv.threshold(redChannel, 240, 255, cv.THRESH_BINARY)

    cv.Canny(laser, 100, 255)
