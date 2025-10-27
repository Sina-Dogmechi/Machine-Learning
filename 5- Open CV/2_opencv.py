import cv2

# print(cv2.__version__)

img = cv2.imread("nature.webp")

print(img.shape)

# only blue
cv2.imshow("my_image", img[:, :, 0])

cv2.waitKey(0)

cv2.destroyAllWindows()