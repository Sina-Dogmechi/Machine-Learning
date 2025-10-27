import cv2

# print(cv2.__version__)

img = cv2.imread("nature.webp")

print(img.shape)

roi = img[50:170, 120:324]

cv2.imshow("my_image", roi)

cv2.waitKey(3000)

cv2.destroyAllWindows()

cv2.imwrite("test.jpg", roi)