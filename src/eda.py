import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pie_chart(data, aspect, labels, title):
    classes_count = data[aspect].value_counts()

    plt.pie(
        classes_count,
        labels=labels,
        autopct="%1.1f%%"
    )

    plt.title(title)
    plt.show()

def bar_graph(data,aspect):
    edu_cnt = data[aspect].value_counts()
    ax = sns.barplot(edu_cnt)
    ax.bar_label(ax.containers[0])


