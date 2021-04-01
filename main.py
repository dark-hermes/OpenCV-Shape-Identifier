import cv2
import json

# Read json files that contains shapes detail
with open('shapes.json') as json_file:
    shapes = json.load(json_file)

# Read a test image and convert that to gray scale
image = cv2.imread('test/someshapes.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show preview image
cv2.imshow('Identifying Shapes', image)
cv2.waitKey(0)

# Find the edge using Canny algorithm
edged = cv2.Canny(gray, 10, 50)
# Find the binary threshold
_, thresh = cv2.threshold(edged, 127, 255, cv2.THRESH_BINARY)

# Find contours of the image with list retrieval
# Chain the points continuously with CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Iterate contours
for contour in contours:

    # Epsilon is accuracy parameter for maximum distance from
    # to approximated contour
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
