import cv2
# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        print("Success! Camera resolution:", frame.shape[1], "x", frame.shape[0])
    cap.release()

# if __name__ == '__main__':
#     toy