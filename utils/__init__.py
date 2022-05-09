# create by andy at 2022/5/7
# reference:
import numpy as np
from typing import List

from spectral import save_rgb

from .Morphology import Morphology


def _any2dec(original_num: str, positional_numeral_system: int) -> int:
    return int(original_num, positional_numeral_system)


def color_code2int_list(color_code: str) -> List[int]:
    assert len(color_code) == 6
    color_list = [_any2dec(color_code[0:2], 16), _any2dec(color_code[2:4], 16), _any2dec(color_code[4:6], 16)]
    return color_list


class Utils:
    @staticmethod
    def to_color(input_gt: np.ndarray,
                 color_codes: List[str] = None,
                 num_classes=None) -> np.ndarray:
        """

        :param color_codes: color corresponding to each class number. the default color code list have five colors.
        :param input_gt: the 2d array
        :param num_classes: the number of categories. if it's None then it equals to the length of color_list
        :return:
        """
        assert isinstance(input_gt, np.ndarray)
        seg_img = np.zeros((input_gt.shape[0], input_gt.shape[1], 3))
        seg_img = seg_img.astype('uint8')
        if color_codes is None:
            color_codes = ["F34B3F", "38d70c", "0c4cd5", "e5e212", "12e5be"]
        if num_classes is None:
            num_classes = len(color_codes)
        color_list = []
        for i in range(len(color_codes)):
            color_list.append(color_code2int_list(color_codes[i]))

        for c in range(num_classes):
            seg_img[:, :, 0] += ((input_gt[:, :] == c) * (color_list[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((input_gt[:, :] == c) * (color_list[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((input_gt[:, :] == c) * (color_list[c][2])).astype('uint8')
        return seg_img

    @staticmethod
    def show(name, array, bands):
        """

        :param name: 要存储的文件名
        :param array: 传入的数组
        :param bands: 要展示的波段，列表类型
        :return:
        """
        save_rgb(name, array, bands)


if __name__ == '__main__':
    pass
