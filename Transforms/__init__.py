# create by andy at 2022/5/9
# reference:
import torch


class Squeeze:
    def __init__(self) -> None:
        pass

    def __call__(self, array: torch.Tensor):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        array = array.squeeze()
        return array

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

if __name__ == '__main__':
    pass
