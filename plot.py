# coding=utf-8
import math
import pickle

import matplotlib.pyplot as plt
import pandas
from numpy import array
from pandas.plotting import register_matplotlib_converters

from learn import exp_window

register_matplotlib_converters()

from lib import normalize, load_data, ALPHA, ALT_SPEED_MEAN, ALT_SPEED_MIN, SPEED_MEAN, SPEED_MIN, ALT_SPEED_MAX, \
    SPEED_MAX, user_events_to_data

user_events_style = {"color": "red", "alpha": 0.33}

(gps, alt, user) = load_data("part_1")

if __name__ == '__main__':
    x = gps['altitude']
    # x = alt["relativeAltitude"]
    smoothed = x.ewm(alpha=ALPHA).mean()
    norm_v_speed = normalize(smoothed.diff(), mean=ALT_SPEED_MEAN, min=ALT_SPEED_MIN, max=ALT_SPEED_MAX)
    norm_speed = normalize(gps['speed'].ewm(alpha=ALPHA).mean(), mean=SPEED_MEAN, min=SPEED_MIN, max=SPEED_MAX)

    normalize(smoothed).plot(label="smoothed")
    # norm_v_speed.plot(label="vertical speed")
    # norm_speed.plot(label="speed")

    # mlp_model = pickle.load(open("./data/model.mlp", 'rb'))
    # i = 0
    # predictions = []
    # for (a, b) in zip(norm_v_speed.iteritems(), norm_speed.iteritems()):
    #     if i % 50 == 0:
    #         (ts, v_speed) = a
    #         (_, speed) = b
    #         if not math.isnan(v_speed) and not math.isnan(speed):
    #             v_speed_feature = exp_window(i, norm_v_speed)
    #             speed_feature = exp_window(i, norm_speed)
    #             x = v_speed_feature + speed_feature
    #             (heli, ski) = mlp_model.predict_proba(array(x).reshape(1, -1))[0]
    #             verdict = mlp_model.predict(array(x).reshape(1, -1))[0]
    #             predictions.append([ts,heli,ski,verdict])
    #     i += 1
    #
    # pd_predictions = pandas.DataFrame(predictions, columns=['timestamp', 'heli', 'ski', 'verdict'])
    # pd_predictions.set_index(['timestamp'], inplace=True)

    # prev_ts = gps.head(1).index[0]
    # for (ts,dt) in pd_predictions.iterrows():
    #     color = 'red' if dt['verdict'] == 0 else 'blue'
    #     plt.axvspan(prev_ts, ts, -1.3, 0.4, color=color, alpha=0.18, linewidth=0.0)
    #     prev_ts = ts

    # for (ts, dt) in user.iterrows():
    #     plt.axvline(ts, -0.4, 2, color="red")
    #     plt.text(
    #         ts, 0, dt['name'].decode("utf-8"), rotation=90,
    #         verticalalignment='bottom', **user_events_style)

    user_data = user_events_to_data(user)
    ys_min = (smoothed).min()
    xs_max = gps.tail(1).index[0]
    for (i, dt) in enumerate(user_data):
        color = 'red' if dt['activity'] == "heli" else 'blue'
        plt.axvspan(dt['from'], dt['to'], -2, 2, color=color,
                    alpha=0.1, linewidth=1.5, picker=True, gid=i + 678)

    plt.legend()
    plt.show()
