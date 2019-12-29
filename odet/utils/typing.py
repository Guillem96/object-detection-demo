from typing import NamedTuple, List


class DetectionInstance(NamedTuple):
    label: str
    box: List[float]