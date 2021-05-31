import pandas as pd
import json

from scipy import stats
from scipy import special

from model_utils import get_label
import numpy as np
import matplotlib.pyplot as plt


def load_stats_df():
    with open('stats.json') as f:
        df = pd.DataFrame(json.load(f)["results"])
    df['correct'] = [True if df.iloc[i]['number'] == df.iloc[i]['id'] else False for i in range(len(df))]
    return df


def score(count, correct_rate, rate):
    if count == 0:
        return 2
    return 2 + (1 / count - correct_rate)*rate


def distribution(rate):
    df = aggregate_df(load_stats_df())
    df['scores'] = [score(df.iloc[i]['count'], df.iloc[i]['correct'], rate) for i in range(len(df))]
    xk = np.arange(len(df))
    pk = special.softmax(df['scores'])
    return stats.rv_discrete(name='custm', values=(xk, pk))


def aggregate_df(df):
    correct_mean = df.groupby(['number']).mean().reset_index()[['number', 'correct']]
    count = pd.DataFrame(df.groupby('number').size()).reset_index()
    count.columns = ['number', 'count']
    aggregated_df = pd.concat([count, correct_mean['correct']], axis=1)
    for i in range(49):
        if i not in aggregated_df['number'].unique():
            tmp = pd.Series(dict(number=i, count=0, correct=0))
            aggregated_df = aggregated_df.append(tmp, ignore_index=True)
    return aggregated_df


def create_bar_chart(ax, y, x, color, inverted=True, title='', xlabel='Accuracy[%]'):
    y_pos = np.arange(len(y))
    ax.barh(y_pos, x, align='center', height=0.5, edgecolor='black', color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y)
    if inverted:
        ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel(xlabel)
    ax.set_title(title)


def calculate_accuracies(df):
    accuracies = []
    length = int(len(df) // 10)
    for i in range(1, length):
        num_correct = 0
        for index, row in df.head(20 * i).iterrows():
            if row['correct']:
                num_correct += 1
        accuracies.append(round(num_correct / (20 * i + 1), 2) * 100)
    return list(range(1, length)), accuracies


def plot_accuracy_over_time(ax, x, y, color, title='Accuracy over time'):
    ax.plot(x, y, color=color)
    ax.set(xlabel='Time', ylabel='Accuracy[%]', title=title)
    ax.grid()
    ax.axes.xaxis.set_ticklabels([])
    ax.set_ylim([0, 100])


def create_plots():
    color_dict = dict(best_train='#91ebe5', worst_train='#f05151', best_test='#2374f7', worst_test='#f01d1d')

    df = load_stats_df()
    df_train = df[df['train_mode'] == True]
    df_test = df[df['train_mode'] == False]
    correct_train = aggregate_df(df_train).sort_values(['correct'], ascending=False)
    correct_test = aggregate_df(df_test).sort_values(['correct'], ascending=False)

    y = [get_label(letter) for letter in list(correct_train.head(5).index)]
    x = list(round(correct_train['correct'].head(5) * 100, 2))

    y2 = [get_label(letter) for letter in list(correct_train.tail(5).index)]
    x2 = list(round((1 - correct_train['correct'].tail(5)) * 100, 2))

    y3 = [get_label(letter) for letter in list(correct_test.head(5).index)]
    x3 = list(round(correct_test['correct'].head(5) * 100, 2))

    y4 = [get_label(letter) for letter in list(correct_test.tail(5).index)]
    x4 = list(round((1 - correct_test['correct'].tail(5)) * 100, 2))

    x5, y5 = calculate_accuracies(df_train)
    x6, y6 = calculate_accuracies(df_test)

    plt.rcdefaults()
    fig, axs = plt.subplots(nrows=3, ncols=2)

    create_bar_chart(ax=axs[0, 0], x=x, y=y, color=color_dict['best_train'], inverted=True,
                     title='Letters with highest accuracy(train)', xlabel='Accuracy[%]')
    create_bar_chart(ax=axs[1, 0], x=x2, y=y2, color=color_dict['worst_train'], inverted=False,
                     title='Letters with highest error(train)', xlabel='Error[%]')
    create_bar_chart(ax=axs[0, 1], x=x3, y=y3, color=color_dict['best_test'], inverted=True,
                     title='Letters with highest accuracy(test)', xlabel='Accuracy[%]')
    create_bar_chart(ax=axs[1, 1], x=x4, y=y4, color=color_dict['worst_test'], inverted=False,
                     title='Letters with highest error(test)', xlabel='Error[%]')
    plot_accuracy_over_time(ax=axs[2, 0], x=x5, y=y5, color='green', title='Accuracy over time(train)')
    plot_accuracy_over_time(ax=axs[2, 1], x=x6, y=y6, color='green', title='Accuracy over time(test)')

    fig.patch.set_facecolor('#f0f0f0')

    plt.tight_layout()
    return fig
