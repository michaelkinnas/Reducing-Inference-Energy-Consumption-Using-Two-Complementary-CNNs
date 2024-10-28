import matplotlib.pyplot as plt
from time import ctime
import csv
import numpy as np

def plot_train_progress(filepath):
    data = {}
    with open(filepath, 'r') as csvfile:
        # reader = csv.reader(csvfile, delimiter=',')
        reader = csv.DictReader(csvfile)
        for key in next(iter(reader)).keys():
            data[key] = []
        for row in reader:
            for k, v in row.items():                
                data[k].append(float(v))

    x_values = [ctime(x)[11:19] for x in data['timestamp']]
    plt.figure(3)
    
    plt.plot(x_values, data['train_loss'], label="Train loss")
    plt.plot(x_values, data['train_acc'], label="Train accuracy")

    # if self.__testloader:
    plt.plot(x_values, data['test_loss'], '--', label="Test loss")
    plt.plot(x_values, data['test_acc'], '--', label="Test accuracy")

    plt.title('Train progress')
    # plt.ylim((0, 1))
    plt.legend()
    plt.ylabel('Percent')
    plt.xlabel('Time')
    plt.xticks(rotation=90)
    plt.show()


def plot_threshold_values(dataframe, dataframe_ps, title, length, flip_x = False):
    scaler = 1
    fig, ax1 = plt.subplots(figsize=(5 * scaler, 4 * scaler))
    fig.set_dpi(100)
    
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


    line1, = ax1.plot(x, y, label="Without post-check")
    line2, = ax1.plot(x_ps, y_ps, label="With post-check")
    ax1.set_ylabel("Accuracy %")
    ax1.set_xlabel("λ parameter")

    # ax1.annotate(text=f'{annot_value:.4f}', xy=(annot_x, annot_y), xytext=(annot_x+0.1, annot_y-0.1), arrowprops=dict(facecolor='black', shrink=0.1))
    ax1.annotate(text=f'{annot_value:.4f}', xy=(annot_x, annot_y))
    ax1.annotate(text=f'{annot_ps_value:.4f}', xy=(annot_ps_x, annot_ps_y))

    line3, = ax2.plot(x, x_b_use, color='C3', label="Second CNN model usage")
    ax2.set_ylabel("Usage %")

    if flip_x:
        plt.gca().invert_xaxis()

    ax2.legend(handles = [line1, line2, line3])
    # ax2.legend(loc=5)


    ax1.yaxis.grid(True, linestyle='--', which='major', color='black', alpha=.25)
    ax1.set_axisbelow(True)
    
    plt.show()


def plot_experiment_results(y1, y2, xtick_labels, y1label, y2label, y1_bracket=None, y2_bracket = None, normalize=False):
    fig, ax1 = plt.subplots()

    if normalize:
        y1 = [x / max(y1) * 100 for x in y1]

    x = np.arange(len(y1))

    width = 0.20
    
    color = 'tab:red'
    # ax1.set_xlabel('softmax threshold')
    ax1.set_ylabel(y1label, color=color)
    ax1.bar(x-(width/2), y1, width, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    if y1_bracket:
        ax1.set_ylim(y1_bracket[0], y1_bracket[1])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    

    color = 'tab:blue'
    ax2.set_ylabel(y2label, color=color)  # we already handled the x-label with ax1
    ax2.bar(x+(width/2), y2, width, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    if y2_bracket:
        ax2.set_ylim(y2_bracket[0], y2_bracket[1])

    plt.xticks(x, xtick_labels)
    # plt.xticks(x, ('a', 'b', 'c', 'd'))
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # fig.set_figwidth(4 * scale)
    # fig.set_figheight(3 * scale)
    # plt.title('resnet20 & resnet56')
    plt.show()



def plot_experiment_results_2(power, acc, preci, recall, f1, labels, xlabel=None, normalize=False, draw_legend=True):

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
    fig.set_dpi(200)

    # mask =  np.tril(heterogeneity_matrix, k=-1)

    mask = np.zeros_like(heterogeneity_matrix, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    data = np.ma.array(heterogeneity_matrix, mask=mask)

    ax.imshow(data, cmap='plasma')
    ax.set_xticklabels(labels=list(heterogeneity_matrix.columns), rotation=90)
    ax.set_xticks(range(len(list(heterogeneity_matrix.columns))))

    ax.set_yticklabels(labels=list(heterogeneity_matrix.index))
    ax.set_yticks(range(len(list(heterogeneity_matrix.index))))
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

  

    for i in range(len(list(heterogeneity_matrix.index))):
        # for j in range(len(list(heterogeneity_matrix.columns))):
        for j in range(i+1):
            if heterogeneity_matrix.iat[i, j] > heterogeneity_matrix.max().iat[0] * 0.5:
                text = ax.text(j, i, f'{heterogeneity_matrix.iat[i, j]:.4f}', ha="center", va="center")
            else:
                text = ax.text(j, i, f'{heterogeneity_matrix.iat[i, j]:.4f}', ha="center", va="center", color="w")

    plt.show()

def plot_heterogeneity_matrix2(heterogeneity_matrix, scale_values=1):
    scale = 0.18 * len(heterogeneity_matrix) # scale as a function of the size of the matrix (number of models in the list)

    fig, ax = plt.subplots()

    fig.set_size_inches((4 * scale, 3 * scale))
    fig.set_dpi(200)

    # mask =  np.tril(heterogeneity_matrix, k=-1)
    
    mask = np.zeros_like(heterogeneity_matrix, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    data = np.ma.array(heterogeneity_matrix, mask=mask)

    ax.imshow(data, cmap='plasma')
   
    ax.set_xticks(range(len(list(heterogeneity_matrix.columns))))
    ax.set_xticklabels([x+1 for x in range(len(list(heterogeneity_matrix.columns)))], rotation=0)
   
    ax.set_yticks(range(len(list(heterogeneity_matrix.index))))
    ax.set_yticklabels([x+1 for x in range(len(list(heterogeneity_matrix.columns)))])
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)


    for i in range(len(list(heterogeneity_matrix.index))):
        # for j in range(len(list(heterogeneity_matrix.columns))):
        for j in range(i+1):
            if heterogeneity_matrix.iat[i, j] > heterogeneity_matrix.max().iat[0] * 0.5:
                # text = ax.text(j, i, f'{heterogeneity_matrix.iat[i, j] * scale:.3f}', ha="center", va="center")
                text = ax.text(j, i, f'{heterogeneity_matrix.iat[i, j] * scale_values:.3f}', ha="center", va="center")
            else:
                text = ax.text(j, i, f'{heterogeneity_matrix.iat[i, j] * scale_values:.3f}', ha="center", va="center", color="w")

    # plt.legend(handles=list(heterogeneity_matrix.columns))

    # from matplotlib.patches import Rectangle
    # extra = 
    extras = [plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0) for x in range(len(list(heterogeneity_matrix.columns)))]
    # extras = [x for x in range(len(list(heterogeneity_matrix.columns)))]



    plt.legend(extras, [f"{i+1}: {j}" for i,j in enumerate(list(heterogeneity_matrix.columns))], frameon=False, ncol=1, fontsize=11.5)
    # plt.gca().invert_yaxis()
    plt.show()

def plot_performance_side_by_side(acc, preci, recall, f1, labels, 
                                         acc2, preci2, recall2, f12, labels2,
                                         xlabel1=None, xlabel2=None, normalize=False, draw_legend=True):

    width = 0.20
    fig_height = 3.5
    scaler = 1.2


    cmap = plt.get_cmap('tab10').colors

    # power.reverse()
    # power2.reverse()
    


    acc.reverse()
    preci.reverse()
    recall.reverse()
    f1.reverse()
    labels.reverse()


    acc2.reverse()
    preci2.reverse()
    recall2.reverse()
    f12.reverse()
    labels2.reverse()


    ind = np.arange(len(acc)) # Number of measurement points (groups)

       

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*scaler, fig_height*scaler))
    
    fig.set_dpi(200)

   

    #LEFT SIDE
    ax1.barh(ind + width * 1.5,    acc,        width, color=cmap[0], label='Accuracy')
    ax1.barh(ind + width * 0.5,    preci,      width, color=cmap[1], label='Precision')
    ax1.barh(ind - width * 0.5,    recall,     width, color=cmap[2], label='Recall')
    ax1.barh(ind - width * 1.5,    f1,         width, color=cmap[3], label='F1')



    # Annotations
    yoffset = -0.055
    xoffset = -0.02
    for i, (a, b, c, d) in enumerate(zip(acc, preci, recall, f1)):
        ax1.annotate(a, xy=(a + xoffset, i + width * 1.5 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(b, xy=(b + xoffset, i + width * 0.5 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(c, xy=(c + xoffset, i - width * 0.5 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(d, xy=(d + xoffset, i - width * 1.5 + yoffset), color='white', horizontalalignment='right')
     
    
    ax1.set(yticks=ind, yticklabels=labels)
    ax1.axis(xmin=0,xmax=1)
    
    # ax.set_yticklabels(ha='center')
    if xlabel1:
        ax1.set_xlabel(xlabel1)

    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)


    

    #RIGHT SIDE
    ax2.barh(ind + width * 1.5,    acc2,        width, color=cmap[0])
    ax2.barh(ind + width * 0.5,    preci2,      width, color=cmap[1])
    ax2.barh(ind - width * 0.5,    recall2,     width, color=cmap[2])
    ax2.barh(ind - width * 1.5,    f12,         width, color=cmap[3])



    for i, (a, b, c, d) in enumerate(zip(acc2, preci2, recall2, f12)):
        ax2.annotate(a, xy=(a + xoffset, i + width * 1.5 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(b, xy=(b + xoffset, i + width * 0.5 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(c, xy=(c + xoffset, i - width * 0.5 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(d, xy=(d + xoffset, i - width * 1.5 + yoffset), color='white', horizontalalignment='right')
    
    ax2.set(yticks=ind, yticklabels=labels2)
    ax2.axis(xmin=0,xmax=1)
    # ax.set_yticklabels(ha='center')
    if xlabel2:
        ax2.set_xlabel(xlabel2)

    ax2.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    

    if draw_legend:
    # ax.legend(loc='center right', bbox_to_anchor=(1.42, 0.5))
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4, fancybox=True, shadow=False)
    
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    plt.show()


def plot_performance_side_by_side_2(acc, preci, recall, f1, labels, 
                                         acc2, preci2, recall2, f12, labels2,
                                         xlabel1=None, xlabel2=None, normalize=False, draw_legend=True):

    width = 0.20
    width2 = 0.155
    fig_height = 4
    scaler = 1.2


    cmap = plt.get_cmap('tab10').colors

    # power.reverse()
    # power2.reverse()
    


    acc.reverse()
    preci.reverse()
    recall.reverse()
    f1.reverse()
    labels.reverse()


    acc2.reverse()
    preci2.reverse()
    recall2.reverse()
    f12.reverse()
    labels2.reverse()


    ind = np.arange(len(acc)) # Number of measurement points (groups)
    ind2 = np.arange(len(acc2))
       

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*scaler, fig_height*scaler))
    
    fig.set_dpi(400)

   

    #LEFT SIDE
    ax1.barh(ind + width * 1.5,    acc,        width, color=cmap[0], label='Accuracy')
    ax1.barh(ind + width * 0.5,    preci,      width, color=cmap[1], label='Precision')
    ax1.barh(ind - width * 0.5,    recall,     width, color=cmap[2], label='Recall')
    ax1.barh(ind - width * 1.5,    f1,         width, color=cmap[3], label='F1')



    # Annotations
    yoffset = -0.055
    xoffset = -0.02
    for i, (a, b, c, d) in enumerate(zip(acc, preci, recall, f1)):
        ax1.annotate(a, xy=(a + xoffset, i + width * 1.5 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(b, xy=(b + xoffset, i + width * 0.5 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(c, xy=(c + xoffset, i - width * 0.5 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(d, xy=(d + xoffset, i - width * 1.5 + yoffset), color='white', horizontalalignment='right')
     
    
    ax1.set(yticks=ind, yticklabels=labels)
    ax1.axis(xmin=0,xmax=1)
    
    # ax.set_yticklabels(ha='center')
    if xlabel1:
        ax1.set_xlabel(xlabel1)

    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)


    

    #RIGHT SIDE
    ax2.barh(ind2 + width2 * 1.5,    acc2,        width2, color=cmap[0])
    ax2.barh(ind2 + width2 * 0.5,    preci2,      width2, color=cmap[1])
    ax2.barh(ind2 - width2 * 0.5,    recall2,     width2, color=cmap[2])
    ax2.barh(ind2 - width2 * 1.5,    f12,         width2, color=cmap[3])



    for i, (a, b, c, d) in enumerate(zip(acc2, preci2, recall2, f12)):
        ax2.annotate(a, xy=(a + xoffset, i + width2 * 1.5 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(b, xy=(b + xoffset, i + width2 * 0.5 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(c, xy=(c + xoffset, i - width2 * 0.5 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(d, xy=(d + xoffset, i - width2 * 1.5 + yoffset), color='white', horizontalalignment='right')
    
    ax2.set(yticks=ind2, yticklabels=labels2)
    ax2.axis(xmin=0,xmax=1)
    # ax.set_yticklabels(ha='center')
    if xlabel2:
        ax2.set_xlabel(xlabel2)

    ax2.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    

    if draw_legend:
    # ax.legend(loc='center right', bbox_to_anchor=(1.42, 0.5))
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4, fancybox=True, shadow=False)
    
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    plt.show()


def plot_efficiency_side_by_side(mah, wh, avg, t95th, t99th, labels, 
                                mah2, wh2, avg2, t95th2, t99th2, labels2,
                                xlabel1=None, xlabel2=None, normalize=False, draw_legend=True):

    cmap = plt.get_cmap('tab10').colors

    mah.reverse()
    wh.reverse()
    avg.reverse()
    t95th.reverse()
    t99th.reverse()
    labels.reverse()


    mah2.reverse()
    wh2.reverse()
    avg2.reverse()
    t95th2.reverse()
    t99th2.reverse()
    labels2.reverse()



    mah_norm = [x / max(mah) for x in mah]
    wh_norm = [x / max(wh) for x in wh]
    avg_norm = [x / max(t99th) for x in avg]
    t95th_norm = [x / max(t99th) for x in t95th]
    t99th_norm = [x / max(t99th) for x in t99th]

    mah2_norm = [x / max(mah2) for x in mah2]
    wh2_norm = [x / max(wh2) for x in wh2]
    avg2_norm = [x / max(t99th2) for x in avg2]
    t95th2_norm = [x / max(t99th2) for x in t95th2]
    t99th2_norm = [x / max(t99th2) for x in t99th2]



    ind = np.arange(len(mah)) # Number of measurement points (groups)
    width = 0.16

    scaler = 1.2




    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*scaler, 4*scaler))

    ax2.xaxis.grid(True, linestyle='--', which='major',
            color='grey', alpha=.25)
    
    ax1.xaxis.grid(True, linestyle='--', which='major',
            color='grey', alpha=.25)
    
    fig.set_dpi(200)

    #LEFT SIDE
    ax1.barh(ind + width * 2,    mah_norm,     width, color=cmap[0], label='Current [mAh]')
    ax1.barh(ind + width * 1,    wh_norm,      width, color=cmap[1], label='Power [Wh]')
    ax1.barh(ind,       avg_norm,     width, color=cmap[2], label='Mean Response Time [ms]')
    ax1.barh(ind - width * 1,    t95th_norm,   width, color=cmap[3], label='95th Tail Latency [ms]')
    ax1.barh(ind - width * 2,    t99th_norm,   width, color=cmap[4], label='99th Tail Latency [ms]')



    # Annotations
    yoffset = -0.055
    xoffset = -0.02
    
    for i, (a, b, c, d, e, at, bt, ct, dt, et) in enumerate(zip(mah, wh, avg, t95th, t99th, mah_norm, wh_norm, avg_norm, t95th_norm, t99th_norm)):
        ax1.annotate(a, xy=(at + xoffset, i + width * 2 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(b, xy=(bt + xoffset, i + width * 1 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(c, xy=(ct + xoffset, i + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(d, xy=(dt + xoffset, i - width * 1 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(e, xy=(et + xoffset, i - width * 2 + yoffset), color='white', horizontalalignment='right')
     
    
    ax1.set(yticks=ind, yticklabels=labels)
    ax1.axis(xmin=0,xmax=1)
    
    # ax.set_yticklabels(ha='center')
    if xlabel1:
        ax1.set_xlabel(xlabel1)




    #RIGHT SIDE
    ax2.barh(ind + width * 2,    mah2_norm,        width, color=cmap[0])
    ax2.barh(ind + width * 1,    wh2_norm,      width, color=cmap[1])
    ax2.barh(ind,               avg2_norm,      width, color=cmap[2])
    ax2.barh(ind - width * 1,    t95th2_norm,     width, color=cmap[3])
    ax2.barh(ind - width * 2,    t99th2_norm,         width, color=cmap[4])



    for i, (a, b, c, d, e, at, bt, ct, dt, et) in enumerate(zip(mah2, wh2, avg2, t95th2, t99th2, mah2_norm, wh2_norm, avg2_norm, t95th2_norm, t99th2_norm)):
        ax2.annotate(a, xy=(at + xoffset, i + width * 2 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(b, xy=(bt + xoffset, i + width * 1 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(c, xy=(ct + xoffset, i + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(d, xy=(dt + xoffset, i - width * 1 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(e, xy=(et + xoffset, i - width * 2 + yoffset), color='white', horizontalalignment='right')
     
    
    ax2.set(yticks=ind, yticklabels=labels2)
    ax2.axis(xmin=0,xmax=1)
    # ax.set_yticklabels(ha='center')
    if xlabel2:
        ax2.set_xlabel(xlabel2)

    if draw_legend:
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=5, fancybox=True, shadow=False)
    

    plt.show()



def plot_response_times_side_by_side(avg, t95th, t99th, labels, 
                                avg2, t95th2, t99th2, labels2,
                                xlabel1=None, xlabel2=None, normalize=False, draw_legend=True):

    cmap = plt.get_cmap('tab10').colors

    width = 0.26
    width2 = 0.20
    fig_height = 3.6
    scaler = 1.2

    avg.reverse()
    t95th.reverse()
    t99th.reverse()
    labels.reverse()

    avg2.reverse()
    t95th2.reverse()
    t99th2.reverse()
    labels2.reverse()

    ind = np.arange(len(t99th)) # Number of measurement points (groups) 
    ind2 = np.arange(len(t99th2))      

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*scaler, fig_height*scaler))
    
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    

    ax2.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    fig.set_dpi(400)

    #LEFT SIDE
    ax1.barh(ind + width * 1,       avg,       width, color=cmap[0], label='Mean Response Time [ms]')
    ax1.barh(ind,                   t95th,     width, color=cmap[1], label='95th Tail Latency [ms]')
    ax1.barh(ind - width * 1,       t99th,     width, color=cmap[2], label='99th Tail Latency [ms]')

    # Annotations
    yoffset = -0.055
    xoffset = -2    
    for i, (a, b, c) in enumerate(zip(avg, t95th, t99th)):
        ax1.annotate(a, xy=(a + xoffset, i + width * 1 + yoffset), color='white', horizontalalignment='right')
        ax1.annotate(b, xy=(b +xoffset, i + yoffset),             color='white', horizontalalignment='right')
        ax1.annotate(c, xy=(c + xoffset, i - width * 1 + yoffset), color='white', horizontalalignment='right')   
    
    ax1.set(yticks=ind, yticklabels=labels)
    # ax1.axis(xmin=0,xmax=1)
    
    # ax.set_yticklabels(ha='center')
    if xlabel1:
        ax1.set_xlabel(xlabel1)

    #RIGHT SIDE
    ax2.barh(ind2 + width2 * 1,    avg2,      width2, color=cmap[0])
    ax2.barh(ind2,               t95th2,      width2, color=cmap[1])
    ax2.barh(ind2 - width2 * 1,    t99th2,     width2, color=cmap[2])

    xoffset = -4
    for i, (a, b, c) in enumerate(zip(avg2, t95th2, t99th2)):
        ax2.annotate(a, xy=(a + xoffset, i + width2 * 1 + yoffset), color='white', horizontalalignment='right')
        ax2.annotate(b, xy=(b + xoffset, i + yoffset),             color='white', horizontalalignment='right')
        ax2.annotate(c, xy=(c + xoffset, i - width2 * 1 + yoffset), color='white', horizontalalignment='right')
   
    ax2.set(yticks=ind2, yticklabels=labels2)
    # ax2.axis(xmin=0,xmax=1)
    # ax.set_yticklabels(ha='center')
    if xlabel2:
        ax2.set_xlabel(xlabel2)

    if draw_legend:
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fancybox=True, shadow=False)
    
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    plt.show()



def plot_power_four_sides(mah, wh, labels, 
                            mah2, wh2, labels2,
                            xlabel1=None, xlabel2=None, normalize=False, draw_legend=True):

    cmap = plt.get_cmap('tab10').colors

    width = 0.65
    width2 = 0.50
    fig_height = 3.5
    scaler = 1.2

    mah.reverse()
    wh.reverse()
    labels.reverse()

    mah2.reverse()
    wh2.reverse()
    labels2.reverse()


    ind = np.arange(len(mah)) # Number of measurement points (groups)
    ind2 = np.arange(len(mah2))
      

    fig, axs = plt.subplots(2, 2, figsize=(10*scaler, fig_height*scaler))
    # plt.gca().set_axisbelow(True)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
     
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    

    ax2.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    

    ax3.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    

    ax4.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)    

    fig.set_dpi(400)

    #TOP LEFT
    ax1.barh(ind,       mah,       width, color=cmap[0], label='Current [mAh]')    
    # Annotations
    yoffset = -0.1
    xoffset = -5
    for i, a in enumerate((mah)):
        ax1.annotate(a, xy=(a + xoffset, i + yoffset), color='white', horizontalalignment='right')
    
    ax1.set(yticks=ind, yticklabels=labels)
    # ax1.axis(xmin=0,xmax=1)
    
    # ax.set_yticklabels(ha='center')
    # if xlabel1:
    #     ax1.set_xlabel(xlabel1)


    #TOP RIGHT 
    ax2.barh(ind2,               mah2,      width2, color=cmap[0])
    
    yoffset = -0.1
    xoffset = -16

    for i, a in enumerate((mah2)):
        ax2.annotate(a, xy=(a + xoffset, i + yoffset),  color='white', horizontalalignment='right')
          
    
    ax2.set(yticks=ind2, yticklabels=labels2)
    # ax2.axis(xmin=0,xmax=1)
    # ax.set_yticklabels(ha='center')
    # if xlabel2:
    #     ax2.set_xlabel(xlabel2)




    #BOTTOM LEFT
    ax3.barh(ind,       wh,       width, color=cmap[1], label='Energy [Wh]')    
    # Annotations
    yoffset = -0.1
    xoffset = -0.02

    for i, a in enumerate((wh)):
        ax3.annotate(a, xy=(a + xoffset, i + yoffset), color='white', horizontalalignment='right')
    
    ax3.set(yticks=ind, yticklabels=labels)
    # ax1.axis(xmin=0,xmax=1)
    
    # ax.set_yticklabels(ha='center')
    if xlabel1:
        ax3.set_xlabel(xlabel1)



    #BOTTOM RIGHT 
    ax4.barh(ind2,               wh2,      width2, color=cmap[1])
        # Annotations
    yoffset = -0.1
    xoffset = -0.06

    for i, a in enumerate((wh2)):
        ax4.annotate(a, xy=(a + xoffset, i + yoffset),  color='white', horizontalalignment='right')
          
    ax4.set(yticks=ind2, yticklabels=labels2)
    # ax2.axis(xmin=0,xmax=1)
    # ax.set_yticklabels(ha='center')
    if xlabel2:
        ax4.set_xlabel(xlabel2)


    if draw_legend:
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5, fancybox=True, shadow=False, frameon=True)

    
    # fig.tight_layout()
    # plt.subplots_adjust(left=0.1,
    #                 bottom=0.1, 
    #                 right=0.9, 
    #                 top=0.9, 
    #                 wspace=0.4, 
    #                 hspace=0.4)

    # ax1.set_xlabel('Current [mAh]')
    # ax2.set_xlabel('Current [mAh]')
    # ax3.set_xlabel('Power [Wh]')
    # ax4.set_xlabel('Power [Wh]')

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax4.set_axisbelow(True)

    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.15, hspace=0.6)
    plt.show()


def plot_memory_lines(big, biglittle, nomemory, dhash, invariants,
                      big2, biglittle2, nomemory2, dhash2, invariants2):

    # big = [225, 302, 396]
    # biglittle = [139, 208, 279]
    # nomemory = [52, 80, 100]
    # dhash = [53, 56, 57]
    # invariants = [68, 75, 85]

    x = [0, 50, 100]

    fig_height = 2.7
    scaler = 1.2

    # ind = np.arange(len(big)) # Number of measurement points (groups)       

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*scaler, fig_height*scaler))
    fig.set_dpi(400)

    ax1.plot(x, big, marker='P', label="C1")
    ax1.plot(x, biglittle, marker='o', label="C2")
    ax1.plot(x, nomemory, marker='^', label="C3")
    ax1.plot(x, dhash, marker='s', label="C4")
    ax1.plot(x, invariants, marker="D", label="C5")

    ax1.set(xticks=x)
    ax1.grid(visible=True, axis='y')
    ax1.set_xlabel("Duplicated samples [%]\n\n(a) CIFAR-10")
    ax1.set_ylabel('Energy [mAh]')
    ax1.legend(ncol=2)
    # plt.legend()
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.43, 1.34), ncol=2, fancybox=True, shadow=False)


    ax2.plot(x, big2, marker='P', label="I1")
    ax2.plot(x, biglittle2, marker='o', label="I2")
    ax2.plot(x, nomemory2, marker='^', label="I3")
    ax2.plot(x, dhash2, marker='s', label="I4")
    ax2.plot(x, invariants2, marker="D", label="I5")

    ax2.set(xticks=x)
    ax2.grid(visible=True, axis='y')
    ax2.set_xlabel("Duplicated samples [%]\n\n(b) ImageNet")
    ax2.set_ylabel('Energy [mAh]')
    # plt.legend()
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=5, fancybox=False, shadow=False)
    ax2.legend(ncol=2)

    plt.show()


def plot_memory_lines_2(big, biglittle, nomemory, dhash, invariants,
                      big2, biglittle2, nomemory2, dhash2, invariants2):

    # big = [225, 302, 396]
    # biglittle = [139, 208, 279]
    # nomemory = [52, 80, 100]
    # dhash = [53, 56, 57]
    # invariants = [68, 75, 85]

    x = [0, 50, 100]

    fig_height = 2.7
    scaler = 1.2

    # ind = np.arange(len(big)) # Number of measurement points (groups)       

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*scaler, fig_height*scaler))
    fig.set_dpi(400)

    ax1.plot(x, big, marker='P', label="C1")
    ax1.plot(x, biglittle, marker='o', label="C4")
    ax1.plot(x, nomemory, marker='^', label="C5")
    ax1.plot(x, dhash, marker='s', label="C6")
    ax1.plot(x, invariants, marker="D", label="C7")

    ax1.set(xticks=x)
    ax1.grid(visible=True, axis='y')
    ax1.set_xlabel("Duplicated samples [%]\n\n(a) CIFAR-10")
    ax1.set_ylabel('Energy [mAh]')
    ax1.legend(ncol=2)
    # plt.legend()
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.43, 1.34), ncol=2, fancybox=True, shadow=False)


    ax2.plot(x, big2, marker='P', label="I1")
    ax2.plot(x, biglittle2, marker='o', label="I4")
    ax2.plot(x, nomemory2, marker='^', label="I5")
    ax2.plot(x, dhash2, marker='s', label="I6")
    ax2.plot(x, invariants2, marker="D", label="I7")

    ax2.set(xticks=x)
    ax2.grid(visible=True, axis='y')
    ax2.set_xlabel("Duplicated samples [%]\n\n(b) ImageNet")
    ax2.set_ylabel('Energy [mAh]')
    # plt.legend()
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=5, fancybox=False, shadow=False)
    ax2.legend(ncol=2)

    plt.show()