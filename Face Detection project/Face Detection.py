import cv2

face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
eye_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

video_cap = cv2.VideoCapture(0)

try:
    while True:
        ret, video_data = video_cap.read()
        if not ret:
            break

        col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

        faces = face_cap.detectMultiScale(col, 1.1, 5, minSize=(30, 30))
        profiles = profile_cap.detectMultiScale(col, 1.1, 5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = col[y:y + h, x:x + w]
            roi_color = video_data[y:y + h, x:x + w]
            eyes = eye_cap.detectMultiScale(roi_gray, 1.1, 5, minSize=(20, 20))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

        for (x, y, w, h) in profiles:
            cv2.rectangle(video_data, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("video_live", video_data)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("a") or key == ord("q") or key == 27:
            break

finally:
    video_cap.release()
    cv2.destroyAllWindows()
    del video_cap
