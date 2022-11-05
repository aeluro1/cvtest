import cv2 as cv
import numpy as np

def resizeImg(image, width = None, height = None, inter = cv.INTER_AREA):
    dim = None # Unnecessary?
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    elif height is not None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return (cv.resize(image, dim, interpolation = inter), dim)

def crop(img, shapes): # This section needs fixing
    if shapes is None:
        return
    maxx = shapes[0]
    mask = np.ones((img.shape[0], img.shape[1]))
    cv.ellipse(mask, maxx, 1, thickness = -1)
   # rect = cv.boundingRect(maxx)
    (cx, cy) = maxx[1] # Unpacks 2-element tuple
    return

def ellipseCirc(ellipse):
    w, h = (i / 2 for i in ellipse[1])
    k = ((h - w) ** 2) / ((h + w) ** 2)
    p = np.pi * (h + w) * (1 + 3 * k / (10 + np.sqrt(4 - 3 * k)))
    return p