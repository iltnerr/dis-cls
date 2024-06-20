import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(output_dir, ep_arr, trainloss_arr, valloss_arr, valacc_arr):
    figloss, ax1 = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    
    ax1.plot(ep_arr, trainloss_arr, 'g', label='Training loss')
    ax1.plot(ep_arr, valloss_arr, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['train','validation'], loc='upper right')
    figloss.savefig(f'{output_dir}/loss.svg')
    plt.close(figloss)

    plt.plot(ep_arr, valacc_arr, 'b', label='Validation accuracy')
    plt.title('Validation accuracy(%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['accuracy'], loc='upper right')
    plt.savefig(f'{output_dir}/acc.svg')
    plt.close()


def plot_learning_curves():
    #TODO
    return


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