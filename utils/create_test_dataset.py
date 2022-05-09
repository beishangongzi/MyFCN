# create by andy at 2022/5/9
# reference:
import os
import shutil

import numpy as np

os.chdir("../data/obt")
ls = np.array(os.listdir("image"))
np.random.shuffle(ls)
ls = ls[:20]
if not  os.path.exists("testImagePreds"):
    os.mkdir("testImagePreds")

if not os.path.exists("testImageMasks"):
    os.mkdir("testImageMasks")
if not os.path.exists("testImage"):
    os.mkdir("testImage")
for l in ls:
    shutil.copy(f"image/{l}", "testImage")
    shutil.copy(f"imageMasks/{l}", "testImageMasks")



if __name__ == '__main__':
    pass
