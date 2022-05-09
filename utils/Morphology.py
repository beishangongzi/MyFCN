# create by andy at 2022/5/9
# reference:
import cv2


class Morphology:
    @staticmethod
    def fun(img, kernel_size, iterations, operation):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        output = cv2.morphologyEx(img, operation, kernel, iterations=iterations)
        return output

    @staticmethod
    def close(img, kernel_size=3, iterations=1):
        return Morphology.fun(img, kernel_size, iterations, cv2.MORPH_CLOSE)

    @staticmethod
    def open(img, kernel_size=3, iterations=1):
        return Morphology.fun(img, kernel_size, iterations, cv2.MORPH_OPEN)

    @staticmethod
    def dilate(img, kernel_size=3, iterations=1):
        return Morphology.fun(img, kernel_size, iterations, cv2.MORPH_DILATE)

    @staticmethod
    def erode(img, kernel_size=3, iterations=1):
        return Morphology.fun(img, kernel_size, iterations, cv2.MORPH_ERODE)
