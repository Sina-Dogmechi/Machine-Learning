import cv2

cap = cv2.VideoCapture("nature_movie.mp4")

while True:

    ret, frame = cap.read()

    if frame is None: break

    cv2.imshow("Frame", frame)

    if cv2.waitKey(30) == ord("q"): break