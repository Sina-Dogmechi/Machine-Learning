import cv2


img = cv2.imread("nature.webp")

# Line
cv2.line(img, (10, 40), (167, 100), (0, 255, 0), 3)

# Rectangle
cv2.rectangle(img, (200, 40), (380, 150), (255, 0, 0), 3)

# Circle
cv2.circle(img, (250, 250), 30, (0, 0, 255), -1)

# Text
cv2.putText(img, 'Machine Learning', (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

cv2.imshow("my_image", img)

cv2.waitKey(0)

cv2.destroyAllWindows()