# coding=utf-8

import matplotlib.pyplot as plt
import pandas
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# exp_window = 1
# linear_window = 2
# mlp = pandas.read_csv('./data/model_runs/model_mlp_1.csv', parse_dates=True)
# print "MAX1: %f" % max(mlp['accuracy']) # 0.855395
#mlp = pandas.read_csv('./data/model_runs/model_mlp_2.csv', parse_dates=True)
# print "MAX2: %f" % max(mlp['accuracy']) # 0.879244
mlp = pandas.read_csv('./data/model_runs/model_mlp_3.csv', parse_dates=True)
# print "MAX2: %f" % max(mlp['accuracy']) # 0.873200
mlp = mlp[mlp['accuracy'] > 0.85]
if __name__ == '__main__':
    # инсайты из первого прогона
    # кажется, linear не работает, только зашумляет данные
    # count = 16, exp 1.45
    # start_second = 45, exp 1.4
    # alpha поменьше = 0.02
    # маленькая alpha требует большего count и start_second
    # const_norm лучше

    # инсайты из второго прогона
    # exp и start_second (1.3, 20) | (?, 100)
    # count = 16 | 70
    # alpha 0.0 - 0.02
    # use_data_perc > 0.5

    # инсайты из третьего прогона
    # alpha 0.01
    # use_data_perc = 0.5

    x = "use_data_perc"
    y = "count"

    grouped = mlp.groupby([x, y])['accuracy'].max()
    grouped = grouped.to_frame().reset_index(level=[x, y])
    grouped.plot(kind="scatter", x=x, y=y,
                 c="accuracy",
                 s=200,
                 colormap='viridis')
    plt.show()
