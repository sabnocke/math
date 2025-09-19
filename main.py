import numpy as np
from numpy_minmax import minmax
from datetime import datetime
from icecream import ic

def minmax_normalize(arr: np.ndarray, inv: bool = True) -> np.ndarray:
    _min, _max = minmax(arr)
    norm = (arr - _min) / (_max - _min)
    return 1 - norm if inv else norm

def dual_minmax_normalize(arr: np.ndarray, scale: tuple[int, int] = (2, 1), inv: bool = False) -> np.ndarray:
    _min, _max = minmax(arr)
    hi, low = scale
    norm = (arr - _min) / (_max - _min)
    norm = hi * norm - low

    return 1 - norm if inv else norm

def robust_scaling(arr: np.ndarray, inv: bool = True) -> np.ndarray:
    q1 = np.quantile(arr, q=0.25)
    q2 = np.quantile(arr, q=0.50)
    q3 = np.quantile(arr, q=0.75)
    norm = (arr - q2) / np.subtract(q3, q1)
    return 1 - norm if inv else norm

def arbit_minmax_normalize(arr: np.ndarray, limit: tuple[int, int] = (0, 1), inv: bool = True) -> np.ndarray:
    _min, _max = minmax(arr)
    a, b = limit
    norm: np.ndarray = a + ((arr - _min) * (b - a)) / (_max - _min)
    return 1 - norm if inv else norm

def gauss_curve(t: np.ndarray, ideal: int = 10, k1: int = 20, k2: int = 50) -> np.ndarray:
    curve = lambda x, y, z: np.exp(-(x - y) ** 2/z)

    more = t[t > ideal]
    # ic(more)
    less = t[t < ideal]
    # ic(curve(more, ideal, k2))
    return np.concatenate((curve(more, ideal, k2), curve(less, ideal, k1)))

def zscore(arr: np.ndarray, inv: bool = True) -> np.ndarray:
    score = (arr - np.mean(arr)) / np.std(arr)

    if inv:
        return score.max() - (score - score.min())
    return score

def regio(weight_a: float = 1/3, weight_b: float = 1/3, weight_c: float = 1/3):
    # TODO add calculation for faraway reservations
    r = np.array([54, 54, 57, 57, 57, 59, 59, 62, 62, 62, 62, 62, 62, 59, 57, 54])
    far_r = np.array([57, 57, 59, 59 , 59, 62, 62, 64, 64, 64, 64, 64, 64, 62, 59, 57])
    r_w = np.array([0, 0, 0.1, 0.1, 0.1, 0.35, 0.5, 0.7, 0.9, 0.6, 0.5, 0.3, 0.1, 0, 0, 0])
    wait = np.array([2, 2, 2, 62, 122, 2, 62, 2, 62, 2, 62, 2, 62, 122, 182, 62], dtype=np.float64)
    wait_2 = np.array([np.inf, 62, 62, np.inf, np.inf, 122, np.inf, 42, np.inf, 122, np.inf, 38, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
    # print(gauss_curve(wait))
    # print(gauss_curve(wait_2))

    total = np.maximum(gauss_curve(wait, k2=1000), gauss_curve(wait_2, k2=1000))
    score = (total - total.mean()) / total.std()
    # ic(score, score + 1, score - np.min(score))
    # ic(total)

    times_r = [(a, 17) for a in range(7, 23)]
    times_r_2 = [datetime.strptime(f"{_t[0]}:{_t[1]}", "%H:%M") for _t in times_r]

    # r_ideal_1 = datetime.strptime("14:00", "%H:%M")
    # r_ideal_2 = datetime.strptime("15:00", "%H:%M")
    r_ideal = datetime.strptime("14:30", "%H:%M")

    def proc(n: datetime):
        first = abs(n - r_ideal_1).total_seconds()
        second = abs(n - r_ideal_2).total_seconds()
        return min(first, second)

    proc_times_r_2 = robust_scaling(np.array([abs(i - r_ideal).total_seconds() for i in times_r_2]), True)
    # ic(proc_times_r_2)
    # ic(arbit_minmax_normalize(np.array([proc(i) for i in times_r_2]), (0, 2)))


    weighted_r = proc_times_r_2
    price = zscore(r)
    price2 = zscore(far_r)
    # weighted_r = proc_times_r_2
    # ic(weighted_r)
    # ic(price)
    ic(score)
    return zip(times_r, weight_a * weighted_r, weight_b * price, weight_c * score, weight_b * price2, r, far_r)

def leo(weight_a: float = 1/3, weight_b: float = 1/3, weight_c: float = 1/3):
    l = np.array([45, 90, 90, 90, 110, 90, 95, 90, 45])
    #TODO add calculation for faraway reservations
    far_l = np.array([45, 50, 65, 85, 95, 85, 90, 90, 45])

    times_l = [
        (10, 34), (11, 34), (13, 34), (15, 34), (16, 34), (18, 34), (19, 34), (21, 34), (23, 34)
    ]
    times_l_2 = [datetime.strptime(f"{_t[0]}:{_t[1]}", "%H:%M") for _t in times_l]
    l_ideal = datetime.strptime("16:00", "%H:%M")
    proc_times_l_2 = robust_scaling(np.array([abs(i - l_ideal).total_seconds() for i in times_l_2]))

    wait = np.array([79, 139, 79, 79, 19, 19, 79, 14, 148])
    score = gauss_curve(wait, k2 = 1000)
    ic(score)
    score = (score - score.mean()) / score.std()
    # ic(score)

    l_w = np.array([0, 0, 0.25, 0.4, 0.8, 0.5, 0.35, 0.1, 0])
    weighted_l = proc_times_l_2
    price = zscore(l * (1 - 0.25))
    price2 = zscore(far_l * (1 - 0.25))
    # ic(price)
    # ic(l)
    # print(l * (1 - 0.25))
    # ic(weighted_l, score, weighted_l + score)
    return zip(times_l, weight_a * weighted_l, weight_b * price, weight_c * score, weight_b * price2, l, far_l)

def main():
    weight = (0.4, 0.1, 0.5)
    max_alt_l, max_alt_l_price = 0, 0
    max_alt_r, max_alt_r_price = 0, 0
    max_r, max_r_price, max_l, max_l_price = 0, 0, 0, 0
    forecast = 10
    result = regio(*weight)
    print("Regiojet")
    for t, a, b, c, d, price, far_price in result:
        h, m = t
        final_score = round(a + b + c, 4)
        final_score2 = round(a + d + c, 4)
        print('\t', f"{h}:{m}", price, round(a, 4), round(b, 4), round(c, 4), round(a + b + c, 4), far_price, final_score2)
        if final_score > max_r:
            max_r = final_score
            max_r_price = price
        if final_score2 > max_alt_r:
            max_alt_r = final_score2
            max_alt_r_price = far_price
    r2 = leo(*weight)
    print("Leo")
    for t, a, b, c, d, price, fa_price in r2:
        h, m = t
        final_score = round(a + b + c, 4)
        final_score_2 = round(a + d + c, 4)
        print('\t', f"{h}:{m}", price, round(a, 4), round(b, 4), round(c, 4), round(a + b + c, 4), fa_price, final_score_2)
        if final_score > max_l:
            max_l = final_score
            max_l_price = price
        if final_score_2 > max_alt_l:
            max_alt_l = final_score_2
            max_alt_l_price = fa_price

    print(max_r, max_alt_r, max_l, max_alt_l)
    print(max_r_price * forecast, max_alt_r_price * forecast, max_l_price * (1 - 0.25) * forecast, max_alt_l_price * (1 - 0.25) * forecast)




if __name__ == "__main__":
    main()

