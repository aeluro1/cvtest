import cv2 as cv
import numpy as np
import matplotlib as plt
import imgprocess as ip

def main():
    cap = cv.imread("flange1.jpg")

    if cap is None:
        print("Unable to load picture")
        exit()


    frame = ip.resizeImg(frame, width = 500)

    frame = cv.cvtColor(cap, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (5, 5), 0)

    cv.imshow("View", frame)

    if cv.waitKey(0) == ord('s'):
        cv.imwrite("test.png", frame)

if __name__ == "__main__":
    main()
