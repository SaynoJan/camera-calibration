import cv2
import numpy as np

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print("Selected:", (x, y))

img = cv2.imread("input.jpg")
clone = img.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image", click)

print("Click 4 points in clockwise order (top-left → top-right → bottom-right → bottom-left)")
while True:
    cv2.imshow("image", clone)
    if cv2.waitKey(1) & 0xFF == 27 or len(points) == 4:
        break

cv2.destroyAllWindows()

pts_src = np.float32(points)
pts_dst = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])

H = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped = cv2.warpPerspective(img, H, (400, 400))

cv2.imwrite("output_aligned.jpg", warped)
cv2.imshow("aligned", warped)
cv2.waitKey(0)
