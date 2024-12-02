import matplotlib.pyplot as plt
import numpy as np


def plot_complementarity_matrix(complementarity_matrix):
    scale = 0.24 * len(complementarity_matrix) # scale as a function of the size of the matrix (number of models in the list)
    fig, ax = plt.subplots()

    fig.set_size_inches((4 * scale, 3 * scale))
    fig.set_dpi(200)

    # mask =  np.tril(heterogeneity_matrix, k=-1)

    mask = np.zeros_like(complementarity_matrix, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    data = np.ma.array(complementarity_matrix, mask=mask)

    ax.imshow(data, cmap='plasma')
    ax.set_xticklabels(labels=list(complementarity_matrix.columns), rotation=90)
    ax.set_xticks(range(len(list(complementarity_matrix.columns))))

    ax.set_yticklabels(labels=list(complementarity_matrix.index))
    ax.set_yticks(range(len(list(complementarity_matrix.index))))
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

  

    for i in range(len(list(complementarity_matrix.index))):
        # for j in range(len(list(heterogeneity_matrix.columns))):
        for j in range(i+1):
            if complementarity_matrix.iat[i, j] > complementarity_matrix.max().iat[0] * 0.5:
                text = ax.text(j, i, f'{complementarity_matrix.iat[i, j]:.4f}', ha="center", va="center")
            else:
                text = ax.text(j, i, f'{complementarity_matrix.iat[i, j]:.4f}', ha="center", va="center", color="w")

    plt.show()