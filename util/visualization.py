import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cycler import cycler
from util.utils import log2dict


def get_attn_weights(stage):
    def rescale(arr):
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = 255 - (arr * 255).astype(np.uint8)
        return arr

    def reshape(arr):
        width = int(arr.shape[0]**0.5)
        return arr.reshape(width, width)
    
    A = stage.A.numpy().squeeze()
    A = rescale(A)
    A = reshape(A)
    return A


def visualize_attention(maps):
    im_size = (256, 256)
    cmap = "plasma"

    fig, axes = plt.subplots(3, 2, figsize=(7,8)) 
    for idx, (title, map) in enumerate(maps.items()):
        weights = cv2.resize(map, im_size, interpolation=cv2.INTER_CUBIC)
        axes.flat[idx].imshow(weights, cmap=cmap)
        axes.flat[idx].set_title(title)
        axes.flat[idx].set_xticks([])
        axes.flat[idx].set_yticks([])
    cbar_ax= fig.add_axes([0.92,0.02,0.02,0.94]) 
    sm=plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(output_dir, save_fig=False):

    # Settings
    colors = cycler('color', ['#EE6666', '#3388BB', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
    SMALL_SIZE = 8
    BIGGER_SIZE = 10
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=BIGGER_SIZE)
    plt.rc('xtick', labelsize=BIGGER_SIZE, direction='out')
    plt.rc('ytick', labelsize=BIGGER_SIZE, direction='out')
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('lines', linewidth=2)
    plt.rcParams['axes.linewidth'] = 0.1    
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='#000000',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('patch', edgecolor='#E6E6E6')

    # Data
    stats = log2dict(f'{output_dir}/log.txt') 
    df = pd.DataFrame.from_dict(stats)

    df_acc = df[[col for col in df.columns if "acc" in col]]
    df_acc = df_acc.div(100)
    df_lr = df[[col for col in df.columns if "lr" in col]]
    df_loss = df[[col for col in df.columns if "loss" in col]].reindex(columns=['test_loss', 'train_loss'])
    df_list = [df_acc, df_lr, df_loss]

    cfg_axes = {
        0: {"ylabel": "Acc@1", "ylimits": [0, 1]},
        1: {"ylabel": "LR", "ylimits": [0, df_lr['lr'].max() * 1.1]},
        2: {"ylabel": "Loss", "ylimits": None}
    }

    # Plots
    fig, axes = plt.subplots(3)
    fig.suptitle("Learning Curves")
    plt.xlabel("Epoch")
    plt.subplots_adjust(hspace=0.5)   

    for idx in range(0, 3):
        df_list[idx].plot(ax=axes[idx])
        axes[idx].get_legend().remove()
        axes[idx].set_ylabel(cfg_axes[idx]["ylabel"])
        axes[idx].set_ylim(cfg_axes[idx]["ylimits"])
        axes[idx].set_xlabel("")

    # Specifics 
    axes[0].legend(['Test'])
    axes[1].get_lines()[0].set_color("black")
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[2].legend(['Test', 'Train'])

    if save_fig:
        plt.savefig(output_dir + "/learning_curves.pdf")
    else:
        plt.show()
