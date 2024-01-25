from scipy.spatial.distance import hamming
from difflib import SequenceMatcher
from typing import Tuple


def _adjust_len(pred: str, target: str) -> Tuple[str, str]:
    if len(pred) < len(target):
        pred = pred.ljust(len(target), " ")
    if len(pred) > len(target):
        pred = pred[: len(target)]
    return (pred, target)


def hamming_dist(pred: str, target: str) -> float:
    pred, target = _adjust_len(pred, target)
    return hamming(list(pred), list(target))


def similarity(pred: str, target: str) -> float:
    pred, target = _adjust_len(pred, target)
    return SequenceMatcher(None, pred, target).ratio()
