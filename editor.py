# coding=utf-8

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas
from matplotlib.widgets import TextBox, Button
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from lib import normalize, load_data, user_events_to_data, KW_GOING_HELI, KW_BACKWARD, widget_log, KW_START

user_events_style = {"color": "red", "alpha": 0.33}
file_name = "small_example"
(gps, alt, user2) = load_data(file_name)
user_marks_holder = [user2]

if __name__ == '__main__':
    x = gps['altitude']
    smoothed = x.ewm(alpha=0.1).mean()
    norm_speed = normalize(gps['speed'].ewm(alpha=0.03).mean())

    charts_ax = normalize(smoothed).plot(label="smoothed")
    norm_speed.plot(label="speed", alpha=0.3)

    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # %Y-%m-%d %H:%M
    error_text = TextBox(plt.axes([0.04, 0.005, 0.8, 0.04]), 'Error', initial="")

    def draw_user_marks():
        for span in draw_user_marks.spans:
            span.remove()
        draw_user_marks.spans = []
        user_data = user_events_to_data(user_marks_holder[0], need_checks=False, expand_time=False)
        for (i, dt) in enumerate(user_data):
            color = 'red' if dt['activity'] == "heli" else 'blue'
            spanid = charts_ax.axvspan(dt['from'], dt['to'], 0.005, 1.0, color=color,
                        alpha=0.1, linewidth=1.5, picker=True, gid=dt['from'])
            draw_user_marks.spans.append(spanid)

    draw_user_marks.spans = []

    draw_user_marks()

    @widget_log(error_text)
    def pick_box(event):
        if pick_box.selected == event.artist:
            pick_box.selected.set_alpha(0.1)
            pick_box.selected = None
            save_btn.label.set_text("Add new")
            return
        if pick_box.selected is not None:
            pick_box.selected.set_alpha(0.1)
        pick_box.selected = event.artist
        ts = get_selected_ts()
        data = user_marks_holder[0].loc[ts]
        mark_type = 'heli' if data['name'] in KW_GOING_HELI else 'ski'
        event.artist.set_alpha(0.4)
        save_btn.label.set_text("Save edit")
        set_state('type', mark_type)
        start_time.set_val(ts)
        plt.draw()


    def get_selected_ts():
        if pick_box.selected is None:
            return None
        else:
            return pick_box.selected.get_gid()


    pick_box.selected = None


    @widget_log(error_text)
    def save_or_add(event):
        new_ts = pandas._libs.tslibs.timestamps.Timestamp(start_time.text)
        mark_type = get_state("type")
        new_type_text = KW_BACKWARD[mark_type]

        if pick_box.selected is None: # creating new mark
            new_df = pandas.DataFrame(
                index=[new_ts],
                columns=["name"],
                data={"name": new_type_text})
            user_marks_holder[0] = user_marks_holder[0].append(new_df, verify_integrity=True, sort=True)
            user_marks_holder[0].index.names = ["timestamp"]
            draw_user_marks()
        else: # edit existing
            old_ts = get_selected_ts()
            user_marks_holder[0].ix[old_ts, 'name'] = new_type_text
            if old_ts != new_ts:
                user_marks_holder[0].rename(index={old_ts: new_ts}, inplace=True)
            draw_user_marks()
        plt.draw()


    @widget_log(error_text)
    def set_state(label, new_state):
        for e in check_btn.params[label]:
            if e['state'] == new_state:
                e['elem'].label.set_fontsize('xx-large')
                e['elem'].label.set_fontweight('bold')
            else:
                e['elem'].label.set_fontsize('small')
                e['elem'].label.set_fontweight('normal')
        check_btn.state_holder[label] = new_state
        plt.draw()


    @widget_log(error_text)
    def check_btn(label, a, a_state, b, b_state):
        check_btn.state_holder = {label: None}
        check_btn.params = {
            label: [
                {"state": a_state, "elem": a},
                {"state": b_state, "elem": b},
            ]
        }
        a.on_clicked(lambda x: set_state(label, a_state))
        b.on_clicked(lambda x: set_state(label, b_state))
    check_btn.state_holder = {}
    check_btn.params = {}


    def get_state(label):
        return check_btn.state_holder[label]


    @widget_log(error_text)
    def finish(event):
        user_marks_holder[0].sort_index()\
            .to_csv('./data/csvs/' + file_name + "_user_eddited.csv")

    # left, bottom, width, height
    start_time = TextBox(plt.axes([0.04, 0.05, 0.3, 0.05]), 'Start', initial="YYYY-mm-dd HH:mm:ss")
    type_heli = Button(plt.axes([0.35, 0.05, 0.18, 0.05]), 'heli')
    type_ski = Button(plt.axes([0.53, 0.05, 0.18, 0.05]), 'ski')
    check_btn("type", type_heli, 'heli', type_ski, 'ski')
    save_btn = Button(plt.axes([0.72, 0.05, 0.27, 0.05]), 'Add new')
    save_btn.on_clicked(save_or_add)
    cid = plt.gcf().canvas.mpl_connect('pick_event', pick_box)
    finish_btn = Button(plt.axes([0.86, 0.005, 0.13, 0.04]), 'Save document')
    finish_btn.on_clicked(finish)

    plt.subplots_adjust(
        left=0.04,
        right=0.99,
        bottom=0.22,
        top=0.99,
        wspace=0.05,
        hspace=0.05)
    plt.show()
