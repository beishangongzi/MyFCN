# create by andy at 2022/5/9
# reference:
import os
import shutil

import numpy as np
def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def mkdir_list(dirs: list):
    for dir_item in dirs:
        mkdir(dir_item)


def move_files(src, dist):
    if not os.path.exists(src):
        return
    for f in os.listdir(src):
        if not f.endswith("npy"):
            continue
        shutil.move(os.path.join(src, f), os.path.join(dist, f))


os.chdir(os.path.join(os.path.dirname(__file__), "..", "data", "obt"))


testImagePreds = "testImagePreds"
testImageMasks = "testImageMasks"
testImage = "testImage"
testImageMorphology = "testImageMorphology"
erode = os.path.join(testImageMorphology, "erode")
dilate = os.path.join(testImageMorphology, "dilate")
open_ = os.path.join(testImageMorphology, "open")
close = os.path.join(testImageMorphology, "close")
image = "image"
imageMask = "imageMasks"
move_files(testImage, image)
move_files(testImageMasks, imageMask)


shutil.rmtree(testImage, ignore_errors=True)
shutil.rmtree(testImageMasks, ignore_errors=True)
shutil.rmtree(testImagePreds, ignore_errors=True)
shutil.rmtree(testImageMorphology, ignore_errors=True)


dirs = [testImagePreds, testImageMasks, testImage,
        testImageMorphology, open_, erode, dilate, close]
mkdir_list(dirs)

ls = np.array(os.listdir(image))
np.random.shuffle(ls)
ls = ls[:15]
for l in ls:
    shutil.move(os.path.join(image, l), testImage)
    shutil.move(os.path.join(imageMask, l), testImageMasks)



if __name__ == '__main__':
    pass
