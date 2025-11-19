import cv2
import numpy as np
import glob
import yaml

# ==== НАСТРОЙКИ ШАХМАТКИ ====
CHECKERBOARD = (7, 5)          # число внутренних углов
square_size = 24.0             # мм (размер клетки)

# ==== ПОДГОТОВКА МИРОВЫХ 3D ТОЧЕК ====
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D точки
imgpoints = []  # 2D точки

# ==== ЗАГРУЗКА ФОТО ====
images = glob.glob("images/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("corners", img)
        cv2.waitKey(300)

cv2.destroyAllWindows()

# ==== КАЛИБРОВКА ====
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# ==== СОХРАНЕНИЕ ====
data = {
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.tolist()
}

with open("camera.yaml", "w") as f:
    yaml.dump(data, f)

print("Saved to camera.yaml")
