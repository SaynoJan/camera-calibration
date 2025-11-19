import cv2
import numpy as np

def intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    if denom == 0:
        return None
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom
    return int(px), int(py)

img = cv2.imread("input.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

lsd = cv2.createLineSegmentDetector(0)
lines = lsd.detect(gray)[0]
lines = [l[0].astype(int) for l in lines]

vertical, horizontal = [], []
for x1, y1, x2, y2 in lines:
    if abs(x1 - x2) < 20:
        vertical.append((x1, y1, x2, y2))
    elif abs(y1 - y2) < 20:
        horizontal.append((x1, y1, x2, y2))

def length(l):
    x1, y1, x2, y2 = l
    return (x2 - x1)**2 + (y2 - y1)**2

vertical = sorted(vertical, key=length, reverse=True)[:2]
horizontal = sorted(horizontal, key=length, reverse=True)[:2]

corners = []
for v in vertical:
    for h in horizontal:
        p = intersection(v, h)
        if p:
            corners.append(p)

corners = np.array(corners)
ordered = np.zeros((4, 2), dtype=np.float32)

s = corners.sum(axis=1)
ordered[0] = corners[np.argmin(s)]
ordered[2] = corners[np.argmax(s)]

d = np.diff(corners, axis=1)
ordered[1] = corners[np.argmin(d)]
ordered[3] = corners[np.argmax(d)]

dst = np.float32([[0, 0], [400, 0], [400, 600], [0, 600]])

H = cv2.getPerspectiveTransform(ordered, dst)
warped = cv2.warpPerspective(img, H, (400, 600))

cv2.imwrite("output_auto.jpg", warped)
cv2.imshow("aligned", warped)
cv2.waitKey(0)
