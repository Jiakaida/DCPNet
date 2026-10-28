import numpy as np


EPS = np.spacing(1)
TYPE = np.float64


def prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    











    gt = gt > 128
    
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    








    return min(2 * matrix.mean(), max_value)
