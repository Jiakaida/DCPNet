import numpy as np
from scipy import ndimage

from .utils import TYPE


class MSIoU:
    def __init__(self):
        











        
        
        self.cell_sizes = np.power(2, np.linspace(0, 9, num=10, dtype=int))
        self.msious = []

    def get_edge(self, mask: np.ndarray):
        




        sx = ndimage.sobel(mask, axis=0, mode="constant")
        sy = ndimage.sobel(mask, axis=1, mode="constant")
        sob = np.hypot(sx, sy)
        sob[sob > 0] = 1
        sob[sob <= 0] = 0
        return sob

    def shrink_by_grid(self, image: np.ndarray, cell_size: int) -> np.ndarray:
        
        h, w = image.shape[:2]

        pad_h = h % cell_size
        if pad_h != 0:
            pad_h = cell_size - pad_h
        pad_w = w % cell_size
        if pad_w != 0:
            pad_w = cell_size - pad_w
        if pad_h != 0 or pad_w != 0:
            image = np.pad(
                image, ((pad_h, 0), (pad_w, 0)), mode="constant", constant_values=0
            )

        h = image.shape[0]
        w = image.shape[1]
        image = image.reshape(h // cell_size, cell_size, w // cell_size, cell_size)
        image = image.sum(axis=(1, 3))
        image[image > 0] = 1
        return image

    def cal_msiou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        pred = self.get_edge(pred)
        gt = self.get_edge(gt)

        ratios = []
        for cell_size in self.cell_sizes:
            s_pred = self.shrink_by_grid(pred, cell_size=cell_size)
            s_gt = self.shrink_by_grid(gt, cell_size=cell_size)
            numerator = np.logical_and(s_pred, s_gt).sum() + 1
            denominator = s_gt.sum() + 1
            ratios.append(numerator / denominator)

        
        msiou = np.trapz(y=ratios, dx=1 / (len(self.cell_sizes) - 1))
        return msiou

    def step(self, pred: np.ndarray, gt: np.ndarray):
        gt = gt > 128
        pred = pred > 128

        msiou = self.cal_msiou(pred, gt)
        self.msious.append(msiou)

    def get_results(self) -> dict:
        



        msiou = np.mean(np.array(self.msious, TYPE))
        return dict(msiou=msiou)
