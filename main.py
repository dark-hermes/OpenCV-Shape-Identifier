import cv2
import json

with open('shapes.json') as json_file:
    shapes = json.load(json_file)

image = cv2.imread('test/someshapes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Identifying Shapes', image)
cv2.waitKey(0)

edged = cv2.Canny(gray, 10, 50)
_, thresh = cv2.threshold(edged, 127, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)

    n_points = len(approx)

    try:
        name = shapes["name"][str(n_points - 3)]
    except KeyError:
        name = ""

    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    if n_points in range(3, 7):
        cv2.drawContours(image, [contour], 0, shapes["color"][str(n_points - 3)], -1)

        if n_points == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) <= 3:
                name = "Square"

    elif 7 <= n_points < 15:
        cv2.drawContours(image, [contour], 0, (255, 0, 0), -1)
        name = "Polygon"

    elif n_points >= 15:
        cv2.drawContours(image, [contour], 0, (255, 0, 196), -1)
        name = "Circle"

    cv2.putText(image, name, (cx - 50 - len(name), cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Identifying Shapes', image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
