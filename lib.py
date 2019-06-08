# coding=utf-8
import datetime
import math
import sys
import time
import traceback
from contextlib import contextmanager
from random import random

import pandas
from matplotlib import pyplot
from numpy import array
from scipy.signal import butter
from scipy.signal.windows import exponential
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# alpha=0.1 даёт ~7 секунд лага после сглаживания
ALPHA = 0.02

# параметры скоростей (м/с), вычислять на большом датасете
ALT_SPEED_MEAN = 0.0
ALT_SPEED_MIN = -8.6
ALT_SPEED_MAX = 61.3
SPEED_MEAN = 8.9
SPEED_MIN = 0.0
SPEED_MAX = 77.5

ONE_HOUR = datetime.timedelta(hours=1)
ZERO_MS = datetime.timedelta(hours=0)


@contextmanager
def timeit_context(name):
    startTime = time.time()
    yield
    elapsedTime = time.time() - startTime
    print('[{}] finished in {} ms'.format(name, int(elapsedTime * 1000)))


def above_zero(arr):
    return [x if x > 0 else 0.00001 for x in arr]


def mult(arr, m):
    return [x * m for x in arr]


def add(arr, m):
    return [x + m for x in arr]


def power(arr):
    transformed, trans_lambda = boxcox(arr)
    return transformed


def gauss(arr):
    numpy_array = array(arr).reshape(len(arr), 1)
    transformer = StandardScaler()
    transformer.fit(numpy_array)
    return [x[0] for x in transformer.transform(numpy_array)]


def normalize(df, mean=None, max=None, min=None):
    if mean is None:
        mean = df.mean()
    if min is None:
        min = df.min()
    if max is None:
        max = df.max()
    return (df - mean) / (max - min)


def normalize_arr(arr):
    numpy_array = array(arr).reshape(len(arr), 1)
    transformer = MinMaxScaler()
    transformer.fit(numpy_array)
    return [x[0] for x in transformer.transform(numpy_array)]


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filtfilt(data, cutoff=100, fs=2000, order=5):
    # b, a = butter_lowpass(cutoff, fs, order=order)
    # y = filtfilt(b, a, data)
    y = exponential(data)
    return y


def load_data(file):
    params = {'index_col': 'timestamp', 'parse_dates': True}
    gps = pandas.read_csv('./data/csvs/%s_location.csv' % file, **params)
    alt = pandas.read_csv('./data/csvs/%s_altitude.csv' % file, **params)
    user = pandas.read_csv('./data/csvs/%s_user.csv' % file, **params)
    return (gps, alt, user)


KW_START = 'Старт от базы'
KW_DOWN = 'Спустились вниз к вертолёту'
KW_UP = 'Вышли из вертолёта на горе'
KW_FINISH = 'Вернулись на базу'
KW_GOING_HELI = (KW_START, KW_DOWN)
KW_GOING_SKI = [KW_UP]
KW_BACKWARD = {
    'heli': 'Спустились вниз к вертолёту',
    'ski':  'Вышли из вертолёта на горе'
}


def user_events_to_data(user, need_checks=True, expand_time=True):
    user = user.sort_index()
    user_data = []
    for (ts, dt) in user.iterrows():
        if len(user_data) == 0:
            if need_checks and dt['name'] != KW_START:
                raise Exception("no start: %s" % dt)

            user_data.append({
                "activity": 'heli' if dt['name'] in KW_GOING_HELI else 'ski',
                "from": ts - (ONE_HOUR if expand_time else ZERO_MS)
            })
        else:
            if need_checks and user_data[-1]['activity'] == "heli" and dt['name'] == KW_DOWN:
                raise Exception("heli-heli")
            if need_checks and user_data[-1]['activity'] == "ski" and dt['name'] == KW_UP:
                raise Exception("ski-ski")

            user_data[-1]['to'] = ts
            if dt['name'] != KW_FINISH:
                user_data.append({
                    "activity": 'heli' if dt['name'] in KW_GOING_HELI else 'ski',
                    "from": ts
                })
    user_data[-1]['to'] = user_data[-1]['to'] + (ONE_HOUR if expand_time else ZERO_MS)
    return user_data


def get_intersecting_ts(ts, from_to_arr):
    for dt in from_to_arr:
        if dt['from'] <= ts < dt['to']:
            return dt
    raise Exception("from to not found")


def widget_log(textbox):
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print traceback.format_exc()
                etype, value, ignore = sys.exc_info()
                textbox.set_val(traceback.format_exception_only(etype, value)[0].replace("\n", " "))
                pyplot.draw()

        return applicator

    return decorate
