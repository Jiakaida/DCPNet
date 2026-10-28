
import warnings

import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

from py_sod_metrics.utils import EPS, TYPE, get_adaptive_threshold, prepare_data


class Fmeasure(object):
    def __init__(self, beta: float = 0.3):
        















        warnings.warn("This class will be removed in the future, please use FmeasureV2 instead!")

        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        





        pred, gt = prepare_data(pred, gt)

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        




        
        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary_predcition = pred >= adaptive_threshold
        area_intersection = binary_predcition[gt].sum()
        if area_intersection == 0:
            adaptive_fm = 0
        else:
            pre = area_intersection / np.count_nonzero(binary_predcition)
            rec = area_intersection / np.count_nonzero(gt)
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)
        return adaptive_fm

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        










        
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)  
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        
        
        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)
        
        
        
        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs
        
        Ps[Ps == 0] = 1
        T = max(np.count_nonzero(gt), 1)
        
        
        precisions = TPs / Ps
        recalls = TPs / T

        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        




        adaptive_fm = np.mean(np.array(self.adaptive_fms, TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=TYPE), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=TYPE), axis=0)  
        recall = np.mean(np.array(self.recalls, dtype=TYPE), axis=0)  
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))

class FmeasureV2:
    def __init__(self, metric_handlers: dict = None):
        




        self._metric_handlers = metric_handlers if metric_handlers else {}

    def add_handler(self, handler_name, metric_handler):
        self._metric_handlers[handler_name] = metric_handler

    @staticmethod
    def get_statistics(binary: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        










        TP = np.count_nonzero(binary[gt])
        FP = np.count_nonzero(binary[~gt])
        FN = FG - TP
        TN = BG - FP
        return {"tp": TP, "fp": FP, "tn": TN, "fn": FN}

    def adaptively_binarizing(self, pred: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        










        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        binary = pred >= adaptive_threshold
        return self.get_statistics(binary, gt, FG, BG)

    def dynamically_binarizing(self, pred: np.ndarray, gt: np.ndarray, FG: int, BG: int) -> dict:
        










        
        pred: np.ndarray = (pred * 255).astype(np.uint8)
        bins: np.ndarray = np.linspace(0, 256, 257)
        tp_hist, _ = np.histogram(pred[gt], bins=bins)  
        fp_hist, _ = np.histogram(pred[~gt], bins=bins)

        
        
        tp_w_thrs = np.cumsum(np.flip(tp_hist))  
        fp_w_thrs = np.cumsum(np.flip(fp_hist))

        
        TPs = tp_w_thrs  
        FPs = fp_w_thrs  
        FNs = FG - TPs  
        TNs = BG - FPs  
        return {"tp": TPs, "fp": FPs, "tn": TNs, "fn": FNs}

    def step(self, pred: np.ndarray, gt: np.ndarray, normalize: bool = True):
        











        if not self._metric_handlers:  
            raise ValueError("Please add your metric handler before using `step()`.")

        if normalize:
            pred, gt = prepare_data(pred, gt)
        else:
            if not (pred.dtype in [np.float32, np.float64] and 0 <= pred.min() <= pred.max() <= 1):
                raise ValueError(
                    "Please make sure the array `pred` is normalized in [0, 1] with dtype float32 or float64."
                )
            if gt.dtype != bool:
                raise ValueError("Please make sure the array `gt` is binary.")

        FG = np.count_nonzero(gt)  
        BG = gt.size - FG  

        dynamical_tpfptnfn = None
        adaptive_tpfptnfn = None
        binary_tpfptnfn = None
        for handler_name, handler in self._metric_handlers.items():
            if handler.dynamic_results is not None:
                if dynamical_tpfptnfn is None:
                    dynamical_tpfptnfn = self.dynamically_binarizing(
                        pred=pred, gt=gt, FG=FG, BG=BG
                    )
                handler.dynamic_results.append(handler(**dynamical_tpfptnfn))

            if handler.adaptive_results is not None:
                if adaptive_tpfptnfn is None:
                    adaptive_tpfptnfn = self.adaptively_binarizing(pred=pred, gt=gt, FG=FG, BG=BG)
                handler.adaptive_results.append(handler(**adaptive_tpfptnfn))

            if handler.binary_results is not None:
                if binary_tpfptnfn is None:
                    
                    binary_tpfptnfn = self.get_statistics(binary=pred > 0.5, gt=gt, FG=FG, BG=BG)
                if handler.sample_based:
                    handler.binary_results.append(handler(**binary_tpfptnfn))
                else:
                    handler.binary_results["tp"] += binary_tpfptnfn["tp"]
                    handler.binary_results["fp"] += binary_tpfptnfn["fp"]
                    handler.binary_results["tn"] += binary_tpfptnfn["tn"]
                    handler.binary_results["fn"] += binary_tpfptnfn["fn"]

    def get_results(self) -> dict:
        




        results = {}
        for handler_name, handler in self._metric_handlers.items():
            res = {}
            if handler.dynamic_results is not None:
                res["dynamic"] = np.mean(np.array(handler.dynamic_results, dtype=TYPE), axis=0)
            if handler.adaptive_results is not None:
                res["adaptive"] = np.mean(np.array(handler.adaptive_results, dtype=TYPE))
            if handler.binary_results is not None:
                if handler.sample_based:
                    res["binary"] = np.mean(np.array(handler.binary_results, dtype=TYPE))
                else:
                    
                    res["binary"] = np.mean(handler(**handler.binary_results))
            results[handler_name] = res
        return results

class MAE(object):
    def __init__(self):
        











        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        





        pred, gt = prepare_data(pred, gt)

        mae = self.cal_mae(pred, gt)
        self.maes.append(mae)

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        




        mae = np.mean(np.abs(pred - gt))
        return mae

    def get_results(self) -> dict:
        




        mae = np.mean(np.array(self.maes, TYPE))
        return dict(mae=mae)


class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        














        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        





        pred, gt = prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        




        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            object_score = self.object(pred, gt) * self.alpha
            region_score = self.region(pred, gt) * (1 - self.alpha)
            sm = max(0, object_score + region_score)
        return sm

    def s_object(self, x: np.ndarray) -> float:
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        score = 2 * mean / (np.power(mean, 2) + 1 + std + EPS)
        return score

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        
        gt_mean = np.mean(gt)
        fg_score = self.s_object(pred[gt]) * gt_mean
        bg_score = self.s_object((1 - pred)[~gt]) * (1 - gt_mean)
        object_score = fg_score + bg_score
        return object_score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        
        h, w = gt.shape
        area = h * w

        
        if np.count_nonzero(gt) == 0:
            cy, cx = np.round(h / 2), np.round(w / 2)
        else:
            
            cy, cx = np.argwhere(gt).mean(axis=0).round()
        
        
        
        cy, cx = int(cy) + 1, int(cx) + 1

        
        w_lt = cx * cy / area
        w_rt = cy * (w - cx) / area
        w_lb = (h - cy) * cx / area
        w_rb = 1 - w_lt - w_rt - w_lb
        score_lt = self.ssim(pred[0:cy, 0:cx], gt[0:cy, 0:cx]) * w_lt
        score_rt = self.ssim(pred[0:cy, cx:w], gt[0:cy, cx:w]) * w_rt
        score_lb = self.ssim(pred[cy:h, 0:cx], gt[cy:h, 0:cx]) * w_lb
        score_rb = self.ssim(pred[cy:h, cx:w], gt[cy:h, cx:w]) * w_rb
        return score_lt + score_rt + score_lb + score_rb

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        




        sm = np.mean(np.array(self.sms, dtype=TYPE))
        return dict(sm=sm)


class Emeasure(object):
    def __init__(self):
        













        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        





        pred, gt = prepare_data(pred=pred, gt=gt)
        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        




        adaptive_threshold = get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        




        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        






        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = (
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        






        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = (
                    2
                    * (combination[0] * combination[1])
                    / (combination[0] ** 2 + combination[1] ** 2 + EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + EPS)
        return em

    def generate_parts_numel_combinations(
        self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel
    ):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        




        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class WeightedFmeasure(object):
    def __init__(self, beta: float = 1):
        














        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        





        pred, gt = prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        
        
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        
        
        E = np.abs(pred - gt)
        
        
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        
        
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        
        
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        
        
        
        
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        
        
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        
        
        
        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + EPS)

        
        Q = (1 + self.beta) * R * P / (R + self.beta * P + EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        


        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        




        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=TYPE))
        return dict(wfm=weighted_fm)
