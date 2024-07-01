import matplotlib.pyplot as plt
import numpy as np

def plot_threshold_values(dataframe, dataframe_ps, title, length, flip_x = False):
    scaler = 0.9
    fig, ax1 = plt.subplots(figsize=(5 * scaler, 4 * scaler))
    ax2 = ax1.twinx()

    x = np.array(dataframe['threshold'])
    y = np.array(dataframe['accuracy']) / length * 100

    x_ps = np.array(dataframe_ps['threshold'])
    y_ps = np.array(dataframe_ps['accuracy']) / length * 100

    x_b_use = np.array(dataframe['model_b_usage']) / length * 100
    # y_b_use_diff_ps = np.array(diff_ps['model_b_usage'])


    vline_x = dataframe.iloc[dataframe['accuracy'].idxmax()].at['threshold']
    annot_x = dataframe.iloc[dataframe['accuracy'].idxmax()].at['threshold']
    annot_y = dataframe.iloc[dataframe['accuracy'].idxmax()].at['accuracy'] / length * 100
    annot_value = dataframe.iloc[dataframe['accuracy'].idxmax()].at['threshold']

    vline_ps_x = dataframe_ps.iloc[dataframe_ps['accuracy'].idxmax()].at['threshold']
    annot_ps_x = dataframe_ps.iloc[dataframe_ps['accuracy'].idxmax()].at['threshold']
    annot_ps_y = dataframe_ps.iloc[dataframe_ps['accuracy'].idxmax()].at['accuracy'] / length * 100
    annot_ps_value = dataframe_ps.iloc[dataframe_ps['accuracy'].idxmax()].at['threshold']


    plt.title(title)

    ax1.axvline(x = vline_x, color = 'C0', linestyle = '--')
    ax1.axvline(x = vline_ps_x, color = 'C1', linestyle = '--')


    line1, = ax1.plot(x, y, label="Without post check")
    line2, = ax1.plot(x_ps, y_ps, label="With postcheck")
    ax1.set_ylabel("Accuracy %")
    ax1.set_xlabel("Threshold hyper-parameter")

    # ax1.annotate(text=f'{annot_value:.4f}', xy=(annot_x, annot_y), xytext=(annot_x+0.1, annot_y-0.1), arrowprops=dict(facecolor='black', shrink=0.1))
    ax1.annotate(text=f'{annot_value:.4f}', xy=(annot_x, annot_y))
    ax1.annotate(text=f'{annot_ps_value:.4f}', xy=(annot_ps_x, annot_ps_y))

    line3, = ax2.plot(x, x_b_use, color='C3', label="Second model usage")
    ax2.set_ylabel("Usage %")

    if flip_x:
        plt.gca().invert_xaxis()

    ax1.legend(handles = [line1, line2, line3])
    # ax2.legend(loc=5)
    plt.show()


def plot_power_results(power, acc, preci, recall, f1, labels, xlabel=None, normalize=False, draw_legend=True):

    cmap = plt.get_cmap('tab10').colors

    power.reverse()
    
    if normalize:
        power_norm = [x / max(power) for x in power]

    acc.reverse()
    preci.reverse()
    recall.reverse()
    f1.reverse()
    labels.reverse()

    ind = np.arange(len(power)) # Number of measurement points (groups)
    width = 0.17

    scaler = 1.2
       

    fig, ax = plt.subplots(figsize=(5*scaler, 4*scaler))
    fig.set_dpi(200)

    ax.barh(ind + width * 2,    power_norm, width, color=cmap[3], label='Energy [mAh]')
    ax.barh(ind + width,        acc,        width, color=cmap[0], label='Accuracy')
    ax.barh(ind,                preci,      width, color=cmap[2], label='Precision')
    ax.barh(ind - width,        recall,     width, color=cmap[1], label='Recall')
    ax.barh(ind - width * 2,    f1,         width, color=cmap[4], label='F1')

    yoffset = -0.055
    xoffset = -0.02
    for i, (a, b, c, d, e, f) in enumerate(zip(power_norm, acc, preci, recall, f1, power)):
        ax.annotate(f, xy=(a + xoffset, i + width * 2 + yoffset), color='white', horizontalalignment='right')
        ax.annotate(b, xy=(b, i), xytext=(b + xoffset, i + width + yoffset), color='white', horizontalalignment='right')
        ax.annotate(c, xy=(c + xoffset, i + yoffset), color='white', horizontalalignment='right')
        ax.annotate(d, xy=(d + xoffset, i - width + yoffset), color='white', horizontalalignment='right')
        ax.annotate(e, xy=(e + xoffset, i - width * 2 + yoffset), color='white', horizontalalignment='right')
        # # break
    
    ax.set(yticks=ind, yticklabels=labels)
    # ax.set_yticklabels(ha='center')
    if xlabel:
        ax.set_xlabel(xlabel)

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.75, pos.height])

    if draw_legend:
    # ax.legend(loc='center right', bbox_to_anchor=(1.42, 0.5))
        ax.legend(loc='upper center', bbox_to_anchor=(0.33, 1.16), ncol=3, fancybox=True, shadow=False)
    plt.show()


def plot_heterogeneity_matrix(heterogeneity_matrix):
    scale = 0.24 * len(heterogeneity_matrix) # scale as a function of the size of the matrix (number of models in the list)
    fig, ax = plt.subplots()

    fig.set_size_inches((4 * scale, 3 * scale))
    fig.set_dpi(100)
    ax.imshow(heterogeneity_matrix, cmap='plasma')
    ax.set_xticklabels(labels=list(heterogeneity_matrix.columns), rotation=90)
    ax.set_xticks(range(len(list(heterogeneity_matrix.columns))))

    ax.set_yticklabels(labels=list(heterogeneity_matrix.index))
    ax.set_yticks(range(len(list(heterogeneity_matrix.index))))
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    for i in range(len(list(heterogeneity_matrix.index))):
        for j in range(len(list(heterogeneity_matrix.columns))):
            if heterogeneity_matrix.iat[i, j] > heterogeneity_matrix.max().iat[0] * 0.5:
                text = ax.text(j, i, f'{heterogeneity_matrix.iat[i, j]:.4f}', ha="center", va="center")
            else:
                text = ax.text(j, i, f'{heterogeneity_matrix.iat[i, j]:.4f}', ha="center", va="center", color="w")

    plt.savefig('./matrix.png')
    plt.show()
    