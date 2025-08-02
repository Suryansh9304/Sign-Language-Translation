import cv2
import numpy as np

def get_gesture(contour):

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    num_vertices = len(approx)

    if num_vertices == 5:
        return "Okay"
    elif num_vertices == 4:
        return "Stop"
    elif num_vertices == 3:
        return "Victory"
    return None

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

lower_color = np.array([0, 100, 100]) 
upper_color = np.array([10, 255, 255]) 

last_gesture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        gesture = get_gesture(largest_contour)
        if gesture and gesture != last_gesture:
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print(f"Detected Gesture: {gesture}")
            last_gesture = gesture 
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()