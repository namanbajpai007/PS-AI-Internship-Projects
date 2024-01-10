import cv2
import time

# Replace 'your_camera_ip' with the actual IP camera URL.
camera_url = 'http://192.168.29.107:8080/video'  # Make sure to specify the correct URL.

cam = cv2.VideoCapture(camera_url)

if not cam.isOpened():
    print("Error: Could not open the camera.")
    exit()

time.sleep(1)

while True:
    ret, img = cam.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        cv2.imshow("cameraFeed", img)
    else:
        print("Error: Invalid frame dimensions.")

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
