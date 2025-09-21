from datetime import datetime
from enum import Enum
from typing import Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray
from numpy_minmax import minmax


def minmax_normalize(arr: NDArray, inv: bool = True) -> NDArray:
    _min, _max = minmax(arr)
    norm = (arr - _min) / (_max - _min)
    return 1 - norm if inv else norm


def sigmoid_range(source: NDArray[np.float64], r1: int = 13, r2: int = 16, scale: int = 2) -> NDArray[np.float64]:
    sigmoid = lambda x: (1 + np.exp(-scale * x)) ** -1

    return sigmoid(source - r1) - sigmoid(source - r2)


def gauss_curve(t: NDArray[np.float64], ideal: int = 10, k1: int = 20, k2: int = 50) -> NDArray[np.float64]:
    curve: Callable[[NDArray[np.float64], int], NDArray[np.float64]] = lambda x, factor: np.exp(
        -(x - ideal) ** 2 / factor)

    return np.where(t > ideal, curve(t, k2), curve(t, k1))


class Type(Enum):
    Leo = 0
    Regio = 1


r_price = np.array([54, 54, 57, 57, 57, 59, 59, 62, 62, 62, 62, 62, 62, 59, 57, 54])
far_r_price = np.array([57, 57, 59, 59, 59, 62, 62, 64, 64, 64, 64, 64, 64, 62, 59, 57])
times_r_2 = [datetime.strptime(f"{_t}:{17}", "%H:%M") for _t in range(7, 23)]
r_wait = np.array([2, 2, 2, 62, 122, 2, 62, 2, 62, 2, 62, 2, 62, 122, 182, 62], dtype=np.float64)
r_wait_2 = np.array(
    [np.inf, 62, 62, np.inf, np.inf, 122, np.inf, 42, np.inf, 122, np.inf, 38, np.inf, np.inf, np.inf, np.inf],
    dtype=np.float64)

l_price = np.array([45, 90, 90, 90, 110, 90, 95, 90, 45])
far_l_price = np.array([45, 50, 65, 85, 95, 85, 90, 90, 45])
times_l = [
    (10, 34), (11, 34), (13, 34), (15, 34), (16, 34), (18, 34), (19, 34), (21, 34), (23, 34)
]
times_l_2 = [datetime.strptime(f"{_t}:34", "%H:%M") for _t in range(10, 24)]
wait_l = np.array([79, 139, 79, 79, 19, 19, 79, 14, 148])

collection: List[Tuple[int, int, int, int, int, int]] = []


def main(weight_dep: float = 1 / 3, weight_price: float = 1 / 3, weight_wait: float = 1 / 3) -> None:
    sale: Callable[[int], int] = lambda x: x - round(x * 0.5)

    for one, two, three, four, five in zip(r_price, far_r_price, times_r_2, r_wait, r_wait_2):
        second: int = three.hour * 100 + three.minute
        collection.append((Type.Regio.value, second, one, two, four, five))
    for one, two, three, four in zip(l_price, far_l_price, times_l_2, wait_l):
        second: int = three.hour * 100 + three.minute
        collection.append((Type.Leo.value, second, sale(one), sale(two), four, 1000))

    complete = np.array(collection, dtype=np.float64)

    hours = complete.T[1] // 100
    hours = sigmoid_range(hours)
    hours = minmax_normalize(hours, False)
    complete.T[1] = hours

    norm1 = minmax_normalize(complete.T[2])
    norm2 = minmax_normalize(complete.T[3])
    complete.T[2] = norm1
    complete.T[3] = norm2

    waiting_min = np.minimum(complete.T[4], complete.T[5])
    waiting_g = gauss_curve(waiting_min, k2=1000)
    complete.T[4] = waiting_g

    weights = np.array([1, weight_dep, weight_price, weight_price, weight_wait, 1])
    new = complete * weights.reshape(1, -1)

    for idx, item in enumerate(new):
        entry = collection[idx]
        price, far_price = entry[2], entry[3]
        hour, minute = divmod(entry[1], 100)
        typ = "Regio" if item[0] == 1 else "Leo"

        print(f"{typ} {hour}:{minute} ({price}, {far_price}) >> {sum(item[1:-1]):.4f}", np.round(item[1:-1], 4),
              sep='\t')


if __name__ == "__main__":
    main()
