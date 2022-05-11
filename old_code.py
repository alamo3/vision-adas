import numpy as np
import cv2


# Not gonna bother documenting this code, don't use it its bad

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    cap = cv2.VideoCapture('test_monkey.hevc')
    car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640,480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 2)
        max_area = 0
        max_x = 0
        max_y = 0
        max_w = 0
        max_h = 0

        for (x, y, w, h) in cars:
            area = (w*h)
            if area >= max_area:
                max_x = x
                max_y = y
                max_h = h
                max_w = w

        cv2.rectangle(frame, (max_x, max_y), (max_x + max_w, max_y + max_h), (0, 0, 255), 2)
        # Display frames in a window
        cv2.imshow('Car Detection', frame)
            # Wait for Enter key to stop
        if cv2.waitKey(16) == 13:
            break

    cv2.destroyAllWindows()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
