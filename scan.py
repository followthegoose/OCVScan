
from transform import four_point_transform
import argparse
import cv2
import imutils

# получаю аргументы из консоли
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "Путь к изображению")
args = vars(ap.parse_args())

# получаю изображение, привожу его к нужным размерам
image = cv2.imread(args["image"])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# применяю фильтр на изображение и ищу точки
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# вывод
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ищу контуры, сохраняя только самые большие
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# поиск контура
for c in cnts:
	# приблизительный контур
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	if len(approx) == 4:
		screenCnt = approx
		break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

# вывод
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# четырёхточечное преобразование
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# вывод
cv2.imshow("Final", imutils.resize(warped, height = 650))
cv2.waitKey(0)

