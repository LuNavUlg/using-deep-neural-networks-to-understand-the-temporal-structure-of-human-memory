import os
import sys
import torch
import pickle
import argparse
import pandas as pd
import numpy as np

#################################
# User input
#################################
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Type of calibration to use", default="fixed")

args = parser.parse_args()

TYPE = str(args.type)

#################################
# Parameters
#################################
NETWORKS = ["alexnet", "resnet18", "efficientnetB0", "efficientnetB4", "efficientnetB7"]
NB_VIDEOS = 64
new_dir = None

wd = os.getcwd()
root_dir = wd + "/"
videos_dir = wd + "/Videos/"

print("Using models: " + str(NETWORKS))
print("Number of videos: " + str(NB_VIDEOS))
print("Calibration type: " + str(TYPE))

#################################
# Define device
#################################
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
print("Using device: " + DEVICE)


#################################
# Calibrate for all layers
#################################
for NETWORK in NETWORKS:
    print(NETWORK)

    if TYPE == "fixed":
        # Define constants by hand and dump
        if NETWORK == "alexnet":
            params = {
                "conv1": {"T_max": 240, "T_min": 0, "tau": 100, "alpha": 50},
                "pool1": {"T_max": 340, "T_min": 100, "tau": 100, "alpha": 50},
                "conv2": {"T_max": 340, "T_min": 100, "tau": 100, "alpha": 50},  # given
                "pool2": {"T_max": 340, "T_min": 100, "tau": 100, "alpha": 50},
                "conv3": {"T_max": 240, "T_min": 0, "tau": 100, "alpha": 50},
                "conv4": {"T_max": 200, "T_min": 0, "tau": 100, "alpha": 50},
                "conv5": {"T_max": 150, "T_min": 50, "tau": 100, "alpha": 50},
                "pool5": {"T_max": 95, "T_min": 20, "tau": 100, "alpha": 50},  # given
                "fc6": {"T_max": 35, "T_min": 5, "tau": 100, "alpha": 50},
                "fc7": {"T_max": 20, "T_min": 5, "tau": 100, "alpha": 50},  # given
                "output": {
                    "T_max": 0.55,
                    "T_min": 0.15,
                    "tau": 100,
                    "alpha": 50,
                },  # given
            }

        elif NETWORK == "resnet18":
            params = {
                "conv1": {"T_max": 750, "T_min": 300, "tau": 100, "alpha": 50},
                "layer1": {"T_max": 150, "T_min": 60, "tau": 100, "alpha": 50},
                "layer2": {"T_max": 60, "T_min": 20, "tau": 100, "alpha": 50},
                "layer3": {"T_max": 30, "T_min": 15, "tau": 100, "alpha": 50},
                "layer4": {"T_max": 110, "T_min": 50, "tau": 100, "alpha": 50},
                "avgpool": {"T_max": 7, "T_min": 2, "tau": 100, "alpha": 50},
                "fc": {"T_max": 20, "T_min": 10, "tau": 100, "alpha": 50},
            }

        elif NETWORK == "efficientnetB0":
            params = {
                "conv1": {"T_max": 340, "T_min": 100, "tau": 100, "alpha": 50},
                "mbconv1": {"T_max": 1500, "T_min": 700, "tau": 100, "alpha": 50},
                "mbconv2": {"T_max": 1200, "T_min": 550, "tau": 100, "alpha": 50},
                "mbconv3": {"T_max": 1000, "T_min": 450, "tau": 100, "alpha": 50},
                "mbconv4": {"T_max": 450, "T_min": 150, "tau": 100, "alpha": 50},
                "mbconv5": {"T_max": 400, "T_min": 150, "tau": 100, "alpha": 50},
                "mbconv6": {"T_max": 300, "T_min": 100, "tau": 100, "alpha": 50},
                "mbconv7": {"T_max": 200, "T_min": 100, "tau": 100, "alpha": 50},
                "conv2": {"T_max": 150, "T_min": 50, "tau": 100, "alpha": 50},
                "pool": {"T_max": 10, "T_min": 0, "tau": 100, "alpha": 50},
                "output": {"T_max": 5, "T_min": 0, "tau": 100, "alpha": 50},
            }

        elif NETWORK == "efficientnetB4":
            params = {
                "conv1": {"T_max": 1200, "T_min": 400, "tau": 100, "alpha": 50},
                "mbconv1": {"T_max": 4000, "T_min": 2000, "tau": 100, "alpha": 50},
                "mbconv2": {"T_max": 4000, "T_min": 2000, "tau": 100, "alpha": 50},
                "mbconv3": {"T_max": 2500, "T_min": 1000, "tau": 100, "alpha": 50},
                "mbconv4": {"T_max": 1300, "T_min": 750, "tau": 100, "alpha": 50},
                "mbconv5": {"T_max": 1000, "T_min": 500, "tau": 100, "alpha": 50},
                "mbconv6": {"T_max": 750, "T_min": 300, "tau": 100, "alpha": 50},
                "mbconv7": {"T_max": 300, "T_min": 200, "tau": 100, "alpha": 50},
                "conv2": {"T_max": 100, "T_min": 50, "tau": 100, "alpha": 50},
                "pool": {"T_max": 5, "T_min": 0, "tau": 100, "alpha": 50},
                "output": {"T_max": 10, "T_min": 0, "tau": 100, "alpha": 50},
            }

        elif NETWORK == "efficientnetB7":
            params = {
                "conv1": {"T_max": 500, "T_min": 200, "tau": 100, "alpha": 50},
                "mbconv1": {"T_max": 1500, "T_min": 900, "tau": 100, "alpha": 50},
                "mbconv2": {"T_max": 2000, "T_min": 1000, "tau": 100, "alpha": 50},
                "mbconv3": {"T_max": 1000, "T_min": 500, "tau": 100, "alpha": 50},
                "mbconv4": {"T_max": 500, "T_min": 250, "tau": 100, "alpha": 50},
                "mbconv5": {"T_max": 450, "T_min": 200, "tau": 100, "alpha": 50},
                "mbconv6": {"T_max": 350, "T_min": 100, "tau": 100, "alpha": 50},
                "mbconv7": {"T_max": 200, "T_min": 75, "tau": 100, "alpha": 50},
                "conv2": {"T_max": 85, "T_min": 50, "tau": 100, "alpha": 50},
                "pool": {"T_max": 7, "T_min": 2, "tau": 100, "alpha": 50},
                "output": {"T_max": 5, "T_min": 0.15, "tau": 100, "alpha": 50},
            }

    elif TYPE == "stats":
        # Load the accumulators (if already computed) (pickles)
        if os.path.exists("videos_l2_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p"):
            with open(
                "videos_l2_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p", "rb"
            ) as f:
                videos_l2 = pickle.load(f)
        else:
            print("Error: Accumulators not found, please compute accumulators first.")
            sys.exit()

        # Compute statistics for each layer for each video
        videos = os.listdir(videos_dir)
        videos.remove(".DS_Store")

        params = {}
        for video in videos:
            params[video] = {}
            layers_hist = videos_l2[video]
            for layer in list(layers_hist.keys()):
                history = layers_hist[layer][
                    2:
                ]  # Remove two first seconds because of big jump

                # Compute statistics
                df = pd.DataFrame(history)
                stats = df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose()

                columns = list(stats.keys())
                values = list(stats.iloc[0])

                params[video][layer] = values

        # Compute mean on all videos for each statistics
        layers = {key: [0] * len(columns) for key in layers_hist.keys()}
        for video in videos:
            for layer in list(params[video].keys()):
                # print(params[video][layer])
                layers[layer] = np.add(layers[layer], params[video][layer])

        for layer in list(layers.keys()):
            layers[layer] /= len(videos)

        # Create dictionaries for each layer
        params = {
            el: {"T_max": 0, "T_min": 0, "tau": 100, "alpha": 50}
            for el in list(layers.keys())
        }

        for layer in list(layers.keys()):
            params[layer]["T_min"] = list(layers[layer])[5]  # '25%'
            params[layer]["T_max"] = list(layers[layer])[7]  # '75%'

    elif TYPE == "cd":
        continue

    else:
        print("Invalid calibration type : ", TYPE)
        sys.exit()

    # Check for any inconsistency
    for layer in list(params.keys()):
        if params[layer]["T_max"] - params[layer]["T_min"] < 0:
            print(NETWORK)
            print("Problem with values T_max and T_min in layer", layer)
            sys.exit()

    # Dump as pickle
    pickle.dump(
        params, open("params_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p", "wb")
    )
    print("Calibration done.")
