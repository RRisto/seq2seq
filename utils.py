import math
import time


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {round(s, 2)}s'

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return f'{as_minutes(s)} - {as_minutes(rs)}'
