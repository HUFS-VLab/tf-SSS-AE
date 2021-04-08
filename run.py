import os
import glob
import json
import time
import shutil
import argparse
import random

import numpy as np

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from plot import save_loss, compare_loss, loss_boxplot
from dataset_loader import DatasetLoader

from model import SSSAE

def train(args, data_list):
    model.train(data_list)
    save_loss(model.loss_list,  args.save_path)


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    # Dataset setting
    parser.add_argument('--train-manifests', nargs='*', default=[], help='-')
    parser.add_argument('--test-manifests', nargs='*', default=[], help='-')
    parser.add_argument('--dataset-name', type=str, default='LSMD', help='-')
    parser.add_argument('--dataset-path', type=str, default='', help='-')
    parser.add_argument('--seq-len', type=int, default=16, help='-')
    parser.add_argument('--dims', type=int, default=80, help='-')
    # Network setting
    parser.add_argument('--n-layers', type=int, default=4, help='-')
    parser.add_argument('--epochs', type=int, default=1, help='-')
    parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True, help='-')
    # Optimizer Setting
    parser.add_argument('--batch-size', type=int, default=None, help='-')
    parser.add_argument('--learning-rate', type=float, default=0.005, help='-')
    parser.add_argument('--beta-1', type=float, default=0.9, help='-')
    parser.add_argument('--beta-2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    # Option
    parser.add_argument('--model-path', type=str, default='./model', help='-')
    parser.add_argument('--model-name', type=str, default='SSS-AE', help='-')
    parser.add_argument('--test', action='store_true', default=False, help='-')
    parser.add_argument('--threshold-weights', type=float, default='2.0', help='-')
    args, unknown = parser.parse_known_args()
    
    
    trainLoader_list = []
    trainData_list = []
    for train_manifest in sorted(args.train_manifests):
        train_loader = DatasetLoader(target_manifest=train_manifest,
                                     dataset_path=args.dataset_path,
                                     is_trained=True)
        trainLoader_list.append(train_loader)
        trainData_list += train_loader.trainData_list

    """ Save path of result """
    model_path = './model'
    model_name = '{}_{}_{}'.format(args.model_name, args.n_layers, args.epochs)

    target_name = ""
    for i, train_loader in enumerate(trainLoader_list):
        if i != len(trainLoader_list) - 1:
            target_name += f"{train_loader.name}_"
        else:
            target_name += f"{train_loader.name}"

    args.save_path   = os.path.join(model_path, args.dataset_name, model_name,  args.dataset_path)
    os.makedirs(args.save_path, exist_ok=True)
    args.save_path   = os.path.join(args.save_path, target_name)

    with tf.Graph().as_default():
        model = SSSAE(args)

        if args.test == False:
            print(f"Target :", target_name, "n_data :", len(trainData_list))

            train(args, trainData_list)
        else:
            testLoader_list = trainLoader_list

            for test_manifest in args.test_manifests:
                test_loader = DatasetLoader(target_manifest=test_manifest, 
                                            dataset_path=args.dataset_path, 
                                            is_trained=False)

                testLoader_list.append(test_loader)


            print("=================================================")
            print(f"Model : SSS-AE(Trained by {target_name} ({len(trainData_list)} samples)")
            print(">>Model loaded")
            print(f"Test datasets: ")

            for test_loader in testLoader_list:
                print(f"{test_loader.name} :\t{len(test_loader.testData_list)} samples")

            # "Directory path of results"
            figure_path      = './figure'
            save_figure_path = os.path.join(figure_path, args.dataset_name, model_name, args.dataset_path)
            os.makedirs(save_figure_path, exist_ok=True)
            save_figure_path = os.path.join(save_figure_path, os.path.basename(args.train_manifests[0]).replace(".json", "_") + str(len(args.train_manifests)))

            model.load_weights()
            print(">>Test...")

            # Evaluate
            loss_dict = {}
            laten_vector_dict = {}

            # calculate mean of l2 loss for train data
            loss_list,  _, _, _ = model.test(trainData_list, target_name)
            loss_dict[f"(Trained){target_name}"] = loss_list.tolist()
            
            avg_loss = np.mean(loss_list)
            threshold = avg_loss * args.threshold_weight

            print(f"Threshold : {threshold}")

            losses_list = []
            label_list = []
            for i, test_loader in enumerate(testLoader_list):
                loss_list, _, n_normal, n_abnormal = model.test(test_loader.testData_list, 
                                                                test_loader.name,
                                                                threshold)
                loss_dict[test_loader.name] = loss_list.tolist()
                laten_vector_dict[test_loader.name] = laten_vector_list.tolist()

                losses_list += list(loss_list)
                label = 1 if i < len(args.train_manifests) else 0
                label_list += [label for i in range(len(loss_list))]

            #loss_boxplot(loss_dict, args.save_path, avg_loss, save_figure_path, weight=threshold_weight)
            #compare_loss(loss_dict, args.save_path, avg_loss, save_figure_path, weight=threshold_weight)
