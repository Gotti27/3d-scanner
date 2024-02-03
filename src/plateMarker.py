import math

import cv2 as cv
import numpy as np

from utils import fit_ellipse_plate, convert_to_polar, get_point_color


def find_plate_elements(frame: np.ndarray, contours, w, h):
    plate = []

    for i, c in enumerate(contours):
        if len(c) < 5:
            continue

        ellipse = cv.fitEllipse(c)

        if 20 < ellipse[1][0] < 60 and 20 < ellipse[1][1] < 60 and \
                20 < ellipse[0][0] < w - 20 and h / 2 < ellipse[0][1] < h:
            flag = True
            center = (round(ellipse[0][0]), round(ellipse[0][1]))

            for p in plate:
                distance = math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)

                if distance < 50:
                    flag = False

            if flag:
                cv.ellipse(frame, ellipse, (0, 255, 0), thickness=5)
                cv.drawMarker(frame, center, (0, 255, 0))
                plate.append(center)

    return plate


def process_plate(plate, frame, original, mtx, dist, debug=False):
    if len(plate) == 0:
        raise Exception("Cannot fit plate with no elements")

    center = fit_ellipse_plate(frame, plate)

    if center is None:
        raise Exception("Cannot fit plate from the elements")

    '''
        if debug:
            cv.ellipse(frame, center, (255, 0, 0), thickness=3)
    '''

    poly = cv.ellipse2Poly((round(center[0][0]), round(center[0][1])),
                           (round(center[1][0] / 2), round(center[1][1] / 2)),
                           round(center[2]), 0, 360, 10)

    plate = list(filter(lambda point: abs(cv.pointPolygonTest(poly, point, True)) < 5, plate))
    symbols = []

    for p in plate:
        color = get_point_color(original.copy(), p)
        if color is None:
            continue

        polars = convert_to_polar(center, p)
        symbols.append([polars[1], color, p])

    symbols.sort(key=lambda s: s[0], reverse=True)
    symbols += symbols
    necklace = 'YWMBMMCCCYWBMYWBYWBC' * 2

    total = [[i, -1] for i in range(len(symbols) // 2)]

    for i in range(len(symbols) // 2):
        votes = []
        for j in range(4):
            if abs(symbols[(i + j)][0] - symbols[(i + j + 1)][0] + 360) % 360 > 25:
                votes = []
                break
            else:
                votes.append(symbols[i + j])

        if len(votes) == 0:
            continue

        temp_offset = necklace.find("".join(map(lambda v: v[1], votes)))

        for j in range(i, i + 5):
            total[j % (len(symbols) // 2)] = [j % (len(symbols) // 2), temp_offset]
            temp_offset += 1
            temp_offset %= 20

    obj_points = []
    pln_points = []

    for i in range(20):
        if i >= len(total):
            break
        if symbols[i] is None:
            continue

        offset = total[i][1]
        if offset == -1:
            continue

        angle = math.radians(18 * offset)
        pln_p = [
            75 * np.cos(angle),
            75 * np.sin(angle)
        ]

        obj_points.append(pln_p)
        pln_points.append(symbols[i][2])

        cv.putText(frame, f"{symbols[i][1]}-{offset}", symbols[i][2], cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (200, 125, 200), 2, cv.LINE_AA)

    if len(obj_points) < 4 or len(pln_points) < 4:
        raise Exception("too few point to calculate plate pose")

    if debug:
        H = cv.findHomography(np.array(pln_points),
                              np.array(
                                  list(map(lambda point: [(point[0] * 2) + 250, (point[1] * 2) + 250], obj_points))))
        transformed = cv.warpPerspective(original, H[0], (500, 500))
        cv.line(transformed, [250, 250], [75 * 2 + 250, 250], (255, 0, 0), 1)
        cv.line(transformed, [250, 250], [250, 75 * 2 + 250], (0, 255, 0), 1)
        cv.line(transformed, [250, 250], [-75 * 2 + 250, 250], (255, 255, 0), 1)
        cv.line(transformed, [250, 250], [250, -75 * 2 + 250], (0, 0, 255), 1)
        cv.drawMarker(transformed, [250, 250], (0, 0, 0), cv.MARKER_TILTED_CROSS, 15, thickness=1)
        cv.imshow('plate homography', transformed)

    obj_points = np.array(obj_points, dtype=np.float32)
    pln_points = np.array(pln_points, dtype=np.float32)
    obj_points = np.append(obj_points, np.zeros((len(obj_points), 1)), axis=1)

    _, r, t = cv.solvePnP(np.array(obj_points, dtype=np.float32), np.array(pln_points, dtype=np.float32),
                          mtx, dist, flags=cv.SOLVEPNP_IPPE)

    projected, _ = cv.projectPoints(np.array([
        [0, 0, 0],
        [75, 0, 0],
        [0, 75, 0],
        [0, -75, 0],
        [-75, 0, 0],
        [0, 0, 75],
    ], dtype=np.float32), r, t, mtx, dist)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[1][0]], (255, 0, 0), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[2][0]], (0, 255, 0), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[3][0]], (0, 0, 255), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[4][0]], (255, 255, 0), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[5][0]], (255, 0, 255), 5)
    cv.drawMarker(frame, [round(i) for i in projected[0][0]], (255, 0, 255), cv.MARKER_STAR)

    return center, r, t
