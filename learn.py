# coding=utf-8
import math
import pickle
import sys
from random import random

import numpy
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

register_matplotlib_converters()

from lib import load_data, user_events_to_data, ALPHA, ALT_SPEED_MEAN, ALT_SPEED_MIN, ALT_SPEED_MAX, \
    SPEED_MAX, SPEED_MIN, SPEED_MEAN, normalize, get_intersecting_ts, timeit_context


def exp_window(current, arr, start_second=20, exp=1.37, count=16):
    # предполагаю, что один тик всегда равен 1 секунде
    lookback = [start_second * math.pow(exp, x) for x in range(0, count)]
    return window(current, arr, lookback)


def linear_window(current, arr, start_second=20, add=60, count=16):
    # предполагаю, что один тик всегда равен 1 секунде
    lookback = [start_second + add * x for x in range(0, count)]
    return window(current, arr, lookback)


def window(current, arr, window_arr):
    result = [arr[current]]
    for lb_s in window_arr:
        i = current - int(lb_s)
        if i < 0 or math.isnan(arr[i]):
            result.append(0.0)
        else:
            result.append(arr[i])
    return result


def preapre_learn_data(gps, alt, user, window_fn, window_params, alpha=ALPHA, const_norm=True, use_data_perc=0.33):
    altitude = gps['altitude']
    speed = gps['speed']
    user_data = user_events_to_data(user)

    # TODO StandartScaler fit on training data
    smoothed = altitude.ewm(alpha=alpha).mean()
    if const_norm:
        norm_v_speed = normalize(smoothed.diff(), mean=ALT_SPEED_MEAN, min=ALT_SPEED_MIN, max=ALT_SPEED_MAX)
        norm_speed = normalize(speed.ewm(alpha=alpha).mean(), mean=SPEED_MEAN, min=SPEED_MIN, max=SPEED_MAX)
    else:
        norm_v_speed = normalize(smoothed.diff())
        norm_speed = normalize(speed.ewm(alpha=alpha).mean())

    X = []
    y = []
    i = 0
    for ((ts, v_speed), (_, speed)) in zip(norm_v_speed.iteritems(), norm_speed.iteritems()):
        if not math.isnan(v_speed) and not math.isnan(speed) and random() <= use_data_perc:
            v_speed_feature = window_fn(i, norm_v_speed, **window_params)
            speed_feature = window_fn(i, norm_speed, **window_params)
            X.append(v_speed_feature + speed_feature)
            u = get_intersecting_ts(ts, user_data)
            y.append(0 if u['activity'] == "heli" else 1)
        i += 1

    return shuffle(X, y, random_state=42)


def score(
        gps_train, alt_train, user_train,
        gps_test, alt_test, user_test,
        window_fn, window_params,
        alpha, const_norm, use_data_perc):
    print "%s, p: %s, alpha: %0.3f, const_norm: %s, use_data_perc: %0.3f" % (
        window_fn.__name__, window_params, alpha, const_norm, use_data_perc
    )

    try:
        (x_train, y_train) = preapre_learn_data(
            gps_train.copy(), alt_train.copy(), user_train.copy(),
            window_fn, window_params,
            alpha, const_norm, use_data_perc)

        (x_test, y_test) = preapre_learn_data(
            gps_test.copy(), alt_test.copy(), user_test.copy(),
            window_fn, window_params,
            alpha, const_norm, use_data_perc)

        n_features = len(x_train[0])
        clf = MLPClassifier(
            solver='adam', alpha=1e-4,
            hidden_layer_sizes=(n_features * 2, n_features * 2),
            random_state=42, max_iter=1200)

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        print "Accuracy score: %s" % accuracy_score(y_test, y_pred)
        return clf
    except:
        print "Unexpected error:", sys.exc_info()[0]


if __name__ == '__main__':
    # alt['relativeAltitude']
    (gps_train, alt_train, user_train) = load_data("part_1")
    (gps_test, alt_test, user_test) = load_data("part_2")

    # X = X1 + X2
    # y = y1 + y2

    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=27)

    experiment_id = 0
    for alpha in (0.001, 0.005, 0.01, 0.015):
        for const_norm in [True]:
            for use_data_perc in (0.5, 0.75):
                for start_second in [80, 100, 120]:
                    for count in [16, 70, 90]:
                        for (wfn, wfn_params) in (
                                (exp_window, {"exp": 1.01}),
                                (exp_window, {"exp": 1.025}),
                                (exp_window, {"exp": 1.05}),
                                (exp_window, {"exp": 1.075}),
                                (exp_window, {"exp": 1.1}),
                                (exp_window, {"exp": 1.15}),
                        ):
                            if "exp" in wfn_params and \
                                    start_second * math.pow(wfn_params['exp'], count) > 3600*4.5:
                                print "exp window more than 4.5h"
                                continue
                            if "add" in wfn_params and \
                                    start_second + wfn_params['add'] * count > 3600*4.5:
                                print "linear window more than 4.5h"
                                continue
                            experiment_id += 1
                            wfn_params['count'] = count
                            wfn_params['start_second'] = start_second
                            print "expid: %d" % experiment_id
                            with timeit_context("score"):
                                model = score(
                                    gps_train, alt_train, user_train,
                                    gps_test, alt_test, user_test,
                                    window_fn=wfn, window_params=wfn_params,
                                    alpha=alpha, const_norm=const_norm, use_data_perc=use_data_perc
                                )


    # функция подбора гиперпараметров
    # grid чототам (paramss)
    # with open('./data/model.mlp', 'wb+') as f:
    #     pickle.dump(model, f)
