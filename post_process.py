# create by andy at 2022/5/9
# reference:
import cv2
import numpy as np

import utils


def process():
    f = "data/obt/testImagePreds/0001_3.npy.png"
    f = cv2.imread(f)
    f = utils.Morphology.close(f, iterations=5)
    print(np.unique(f))
    cv2.imwrite("s.png", f)


if __name__ == '__main__':
    process()
