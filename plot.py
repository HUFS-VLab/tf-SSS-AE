import os
import glob
import json
import numpy as np

import matplotlib.pyplot as plt


def save_loss(loss_list, model_path):
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(loss_list, color='blue', linestyle="-", label="loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig(f"{model_path}/loss.png")
    plt.close()
    
    
#def compare_loss(loss_dict, model_path, value, save_figure_path, weight): 

#def loss_boxplot(loss_dict, model_path, value, save_figure_path, weight):
