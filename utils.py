import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

from PIL import Image
from torchvision import transforms
from torchvision import models


def retrieve_videos(video_path, number_of_videos=10, video_format=".avi"):
    """
    This function retrieves the videos in the video directory and returns
    a list of the videos.

    Args:
      video_path : path to the directory containing the videos
      number_of_videos : number of videos to retrieve (default=10)
      video_format : format of the videos (default=.avi)

    Returns:
      videos : list of the videos names in the directory
    """
    videos = []
    for video in os.listdir(video_path):
        if video.endswith(video_format):
            if number_of_videos == 0:
                break
            videos.append(video)
            number_of_videos -= 1
    return videos


def show_classification(output):
    """
    This function shows the top 5 predictions for the input image fed to the classification network.
    Args:
        output: the output of the classification network
    Returns:
        None, but prints the top 5 predictions
    """

    # map the class no to the corresponding class name
    with open("imagenet_classes.txt", "r") as labels:
        classes = [line.strip() for line in labels.readlines()]

    # print the top 5 predictions
    print("Top 5 predictions:")
    for i in range(5):
        print("class " + str(i) + ": " + str(classes[i]))

    # sort the probabilities in descending order
    sorted, indices = torch.sort(output, descending=True)
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    # obtain the top 5 predictions
    results = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
    print("\n print the top 5 predictions")
    for i in range(5):
        print("{:.2f}%: {}".format(results[i][1], results[i][0]))


def get_layers(network: str, type: str) -> list:
    """
    This function returns the layers of the network of which the name is given as input.
    Args:
        network: the name of the network
        type: str indicating whether the run should be naive, lasso or vanilla. If type is None,
        all the layers will be retrieved.
    Returns:
        layers: the list of layers of the network
    """
    # Get the layers
    if type == "vanilla":
        if network == "alexnet":
            layers = ["conv2", "pool5", "fc7", "output"]

    else:
        # Get all the layers
        if network == "alexnet":
            layers = [
                "conv1",
                "pool1",
                "conv2",
                "pool2",
                "conv3",
                "conv4",
                "conv5",
                "pool5",
                "fc6",
                "fc7",
                "output",
            ]
        elif network == "resnet18":
            layers = [
                "conv1",
                "layer1",
                "layer2",
                "layer3",
                "layer4",
                "avgpool",
                "fc",
            ]

        elif network.startswith("efficientnet"):
            layers = [
                "conv1",
                "mbconv1",
                "mbconv2",
                "mbconv3",
                "mbconv4",
                "mbconv5",
                "mbconv6",
                "mbconv7",
                "conv2",
                "pool",
                "output",
            ]

    return layers


def transform(network: str, frame: np.ndarray) -> torch.Tensor:
    """
    This function transforms the input frame to a tensor that can be fed to the network.
    Args:
        network: the name of the network
        frame: the input frame
    Returns:
        processed_frame: the processed frame
    """
    if network == "resnet18" or "alexnet":
        dtransform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    elif network.startswith("efficientnet"):
        dtransform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    # Convert to PIL
    img = Image.fromarray(frame)
    processed_frame = dtransform(img)

    return processed_frame


def get_params(type: str, network: str, layers: list) -> dict:
    """
    This function the parameters decided for each layer of the network.
    Args:
        type: str indicating whether doing vanilla, naive or lasso execution. If None, get all the layers parameters
        network: the name of the network
        layers: layers to take into account
    Returns:
        params: the parameters for each layer
    """
    NB_VIDEOS = 64

    # Load complete dictionary
    if os.path.exists("params_" + str(NB_VIDEOS) + "_" + str(network) + ".p"):
        with open("params_" + str(NB_VIDEOS) + "_" + str(network) + ".p", "rb") as f:
            params = pickle.load(f)
    else:
        print("No data")

    if type == "vanilla":
        if network == "alexnet":
            layers = ["conv2", "pool5", "fc7", "output"]

    params = {key: params[key] for key in layers}

    return params


def plot_change_detection(
    layers, l2, thresholds, params, results_dir, video, frequency=30
):
    """
    This function plots the change detection for each layer.
    Args:
        layers: the list of layers
        l2: the list of l2 norms
        thresholds: the list of thresholds
        frequency: the frequency of the video
        results_dir: the directory in which plot is saved
        video: the name of the corresponding video
    Returns:
        None, but plots the change detection
    """
    # Plot with SNS
    sns.set()
    plt.figure(figsize=(25, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    nb = len(layers)
    rows = int(nb / 2) + 1
    for i, layer in enumerate(layers):
        # add a new subplot iteratively
        ax = plt.subplot(rows, 2, i + 1)
        time = np.arange(len(thresholds[layer])) * frequency

        ax.plot(time, l2[layer], label="Euclidean norm")
        ax.plot(time, thresholds[layer], label="Attention Threshold")
        # Make an horizontal line at T_max for each layer
        ax.axhline(
            y=params[layer]["T_max"],
            color="r",
            linestyle="--",
            label="Maximum threshold",
            alpha=0.5,
        )

        # Make an horizontal line at T_min for each layer
        ax.axhline(
            y=params[layer]["T_min"],
            color="g",
            linestyle="--",
            label="Minimum threshold",
            alpha=0.5,
        )

        # Put dots at each timestep on the curves
        ax.scatter(time, thresholds[layer], s=3, color="r")
        ax.scatter(time, l2[layer], s=3, color="b")
        ax.set_title("Change detection in layer  : " + layer)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend()

    plt.savefig(results_dir + "change_detection_" + str(video) + ".jpeg")
    # plt.show()


def plot_sense_of_time(layers, accumulators_history, results_dir, video, frequency=30):
    """
    This function plots the sense of time for each layer
    (i.e. the value of the accumulator for each layer).
    Args:
        layers: the list of layers
        accumulators_history: the list of accumulators
        frequency: the frequency of the video
        results_dir: the directory in which plot is saved
        video: the name of the corresponding video
    Returns:
        None, but plots the sense of time
    """
    # Plot with SNS
    sns.set()
    plt.figure(figsize=(25, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    nb = len(layers)
    rows = int(nb / 2) + 1
    for i, layer in enumerate(layers):
        # add a new subplot iteratively
        ax = plt.subplot(rows, 2, i + 1)
        time = np.arange(len(accumulators_history[layer])) * frequency

        ax.plot(time, accumulators_history[layer], label="Accumulator")
        ax.set_title("Sense of time in layer  : " + layer)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend()

    plt.savefig(results_dir + "sense_of_time_" + str(video) + ".jpeg")
    # plt.show()


def read_excel(
    root_dir,
    videos_dir,
    filename="TCSE_3060_Comport_Last.xlsx",
    columns=[
        "Participant",
        "Trial_Nb",
        "BlackScreen_Duration",
        "Duration_Segment",
        "Stimuli_Name",
    ],
):
    df = pd.read_excel(root_dir + filename)

    # Only keep interesting columns
    # 'Participant' : participant number
    # 'Trial_nb' : trial number
    # 'BlackScreen_Duration' : duration of the black screen
    # 'Duration_Segment' : duration of the segment/video
    # 'Stimuli_Name' : name of the stimuli/video
    df = df[columns]

    # Convert the duration of the black screen and the segment to seconds
    df = df.assign(
        BlackScreen_Duration=df["BlackScreen_Duration"].apply(lambda x: x / 1000)
    )

    # Each line corresponds to a video, check if all the videos are in the directory
    videos = df["Stimuli_Name"].unique()
    if len(videos) != len(os.listdir(videos_dir)):
        print(
            "Number of videos in the directory : {}".format(len(os.listdir(videos_dir)))
        )
        print("Number of videos in the dataframe : {}".format(len(videos)))

    return df


def build_duration_dataset(
    videos_dir,
    df,
    number_of_videos=64,
    video_format=".avi",
):
    """
    This function builds the dataset used for real-duration baseline for the training of the model.
    Args:
        videos_dir: the directory containing the videos
        df: the dataframe containing the duration of the black screen for each video (value to predict)
        number_of_videos: the number of videos to use for the training
        video_format: the format of the videos
    Returns:
        X_train: the training set
        y_train: the labels of the training set
        patients_train: the patients corresponding to the training set
        stimuli_names_train: the stimuli names corresponding to the training set
        X_test: the test set
        y_test: the labels of the test set
        patients_test: the patients corresponding to the test set
        stimuli_names_test: the stimuli names corresponding to the test set
    """
    # Get all videos
    videos = retrieve_videos(
        videos_dir, number_of_videos=number_of_videos, video_format=video_format
    )

    # Select videos for testing and the remainder for training
    random.seed(42)
    test_videos = random.choices(videos, k=8)
    train_videos = [x for x in videos if x not in test_videos]

    # We want to predict the duration of the black screen
    # Store these in a list : y_train
    X_train = []
    y_train = []
    patients_train = []
    stimuli_names_train = []

    X_test = []
    y_test = []
    patients_test = []
    stimuli_names_test = []

    print("Generating the training set for videos : ")
    for video in videos:
        # WARNING : the videos will not be in the df because the names are not the same !!!
        # We must remove the ".avi" at the end of the video name
        # And we must also remove the remaining "." in the name
        video_name = video  # save the original name before changing it
        video = video.replace(".avi", "")
        video = video.replace(".", "")

        # Some videos might have been used in several trials
        # Me must create an entry for each trial !!!

        # Get the trials related to the video : duration of the black screen and participant ID
        real_durations = df[df["Stimuli_Name"] == video]["Duration_Segment"].values
        durations = df[df["Stimuli_Name"] == video]["BlackScreen_Duration"].values
        ids = df[df["Stimuli_Name"] == video]["Participant"].values

        for real_duration, duration, id in zip(real_durations, durations, ids):
            if video_name in test_videos:  # Test set
                X_test += [[real_duration]]
                y_test += [duration]
                patients_test += [id]
                stimuli_names_test += [video]
            else:  # Training set
                X_train += [[real_duration]]
                y_train += [duration]
                patients_train += [id]
                stimuli_names_train += [video]

    print(
        "TRAINING SET : contains {} samples, {} patients based on {} videos".format(
            len(X_train), len(set(patients_train)), len(train_videos)
        )
    )

    print(
        "TEST SET : contains {} samples, {} patients based on {} videos".format(
            len(X_test), len(set(patients_test)), len(test_videos)
        )
    )

    return (
        X_train,
        y_train,
        patients_train,
        stimuli_names_train,
        X_test,
        y_test,
        patients_test,
        stimuli_names_test,
    )


def build_dataset(
    videos_dir,
    videos_accs,
    df,
    number_of_videos=64,
    video_format=".avi",
):
    """
    This function builds the dataset for the training of the model.
    Args:
        videos_dir: the directory containing the videos
        videos_accs: the dictionary containing the accumulators for each layer
        df: the dataframe containing the duration of the black screen for each video (value to predict)
        number_of_videos: the number of videos to use for the training
        video_format: the format of the videos

    Returns:
        X_train: the training set
        y_train: the labels of the training set
        patients_train: the patients corresponding to the training set
        stimuli_names_train: the stimuli names corresponding to the training set
        X_test: the test set
        y_test: the labels of the test set
        patients_test: the patients corresponding to the test set
        stimuli_names_test: the stimuli names corresponding to the test set
    """
    # For each video, we must predict the duration of the black screen
    # Each participant has a unique ID which we'll use to perform a K-fold cross validation

    # Get all videos
    videos = retrieve_videos(
        videos_dir, number_of_videos=number_of_videos, video_format=video_format
    )

    # Select videos for testing and the remainder for training
    random.seed(42)
    test_videos = random.choices(videos, k=8)
    train_videos = [x for x in videos if x not in test_videos]

    # Create an array containing the participants IDs, we'll use it to perform a K-fold cross validation : X_train
    # And add the accumulators for each layer

    # We want to predict the duration of the black screen
    # To perform the cross validation, we'll use the participants IDs too
    # Store these in a list : y_train

    X_train = []
    y_train = []
    patients_train = []
    stimuli_names_train = []

    X_test = []
    y_test = []
    patients_test = []
    stimuli_names_test = []

    print("Generating the training set for videos : ")
    # print(videos_accs)

    for video in videos:
        if video not in videos_accs.keys():
            continue

        accs = videos_accs[video]
        # Transform the dictionary into a list
        accs = [accs[layer] for layer in accs.keys()]

        # WARNING : the videos will not be in the df because the names are not the same !!!
        # We must remove the ".avi" at the end of the video name
        # And we must also remove the remaining "." in the name
        video_name = video  # save the original name before changing it
        video = video.replace(".avi", "")
        video = video.replace(".", "")

        # Some videos might have been used in several trials
        # Me must create an entry for each trial !!!

        # Get the trials related to the video : duration of the black screen and participant ID
        durations = df[df["Stimuli_Name"] == video]["BlackScreen_Duration"].values
        ids = df[df["Stimuli_Name"] == video]["Participant"].values

        for duration, id in zip(durations, ids):
            if video_name in test_videos:  # Test set
                X_test += [accs]
                y_test += [duration]
                patients_test += [id]
                stimuli_names_test += [video]
            else:  # Training set
                X_train += [accs]
                y_train += [duration]
                patients_train += [id]
                stimuli_names_train += [video]

    print(
        "TRAINING SET : contains {} samples, {} patients based on {} videos".format(
            len(X_train), len(set(patients_train)), len(train_videos)
        )
    )

    print(
        "TEST SET : contains {} samples, {} patients based on {} videos".format(
            len(X_test), len(set(patients_test)), len(test_videos)
        )
    )

    return (
        X_train,
        y_train,
        patients_train,
        stimuli_names_train,
        X_test,
        y_test,
        patients_test,
        stimuli_names_test,
    )


def get_model(name):
    if name == "resnet18":
        model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    elif name == "alexnet":
        model = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
    elif name == "efficientnetB0":
        model = models.efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1")
    elif name == "efficientnetB4":
        model = models.efficientnet_b4(weights="EfficientNet_B4_Weights.IMAGENET1K_V1")
    elif name == "efficientnetB7":
        model = models.efficientnet_b7(weights="EfficientNet_B7_Weights.IMAGENET1K_V1")
    else:
        print("Invalid network name")

    return model


def hook_layers(model, network, type, layers):
    """
    This function hooks the layers of the model to retrieve the activations.

    Args:
        model: the model to hook
        network: the network used
        type: the type of hooking (vanilla or layers)
        layers: the layers to hook

    Returns:
        activation: the activations of the model
    """
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    children = list(model.children())
    if type == "vanilla":  # hook layers mentioned in article
        if network == "alexnet":
            children[0][3].register_forward_hook(get_activation("conv2"))
            children[0][5].register_forward_hook(get_activation("pool5"))
            children[2][4].register_forward_hook(get_activation("fc7"))
            children[2][6].register_forward_hook(get_activation("output"))

    else:  # hook all layers
        for layer in layers:
            if network == "alexnet":
                if layer == "conv1":
                    children[0][0].register_forward_hook(get_activation(layer))
                elif layer == "pool1":
                    children[0][2].register_forward_hook(get_activation(layer))
                elif layer == "conv2":
                    children[0][3].register_forward_hook(get_activation(layer))
                elif layer == "pool2":
                    children[0][5].register_forward_hook(get_activation(layer))
                elif layer == "conv3":
                    children[0][6].register_forward_hook(get_activation(layer))
                elif layer == "conv4":
                    children[0][8].register_forward_hook(get_activation(layer))
                elif layer == "conv5":
                    children[0][10].register_forward_hook(get_activation(layer))
                elif layer == "pool5":
                    children[0][12].register_forward_hook(get_activation(layer))
                elif layer == "fc6":
                    children[2][1].register_forward_hook(get_activation(layer))
                elif layer == "fc7":
                    children[2][4].register_forward_hook(get_activation(layer))
                elif layer == "output":
                    children[2][6].register_forward_hook(get_activation(layer))

            elif network == "resnet18":
                if layer == "conv1":
                    children[0].register_forward_hook(get_activation(layer))
                elif layer == "layer1":
                    children[4].register_forward_hook(get_activation(layer))
                elif layer == "layer2":
                    children[5].register_forward_hook(get_activation(layer))
                elif layer == "layer3":
                    children[6].register_forward_hook(get_activation(layer))
                elif layer == "layer4":
                    children[7].register_forward_hook(get_activation(layer))
                elif layer == "avgpool":
                    children[-2].register_forward_hook(get_activation(layer))
                elif layer == "fc":
                    children[-1].register_forward_hook(get_activation(layer))

            elif network.startswith("efficientnet"):
                if layer == "conv1":
                    children[0][0].register_forward_hook(get_activation(layer))
                elif layer == "mbconv1":
                    children[0][1].register_forward_hook(get_activation(layer))
                elif layer == "mbconv2":
                    children[0][2].register_forward_hook(get_activation(layer))
                elif layer == "mbconv3":
                    children[0][3].register_forward_hook(get_activation(layer))
                elif layer == "mbconv4":
                    children[0][4].register_forward_hook(get_activation(layer))
                elif layer == "mbconv5":
                    children[0][5].register_forward_hook(get_activation(layer))
                elif layer == "mbconv6":
                    children[0][6].register_forward_hook(get_activation(layer))
                elif layer == "mbconv7":
                    children[0][7].register_forward_hook(get_activation(layer))
                elif layer == "conv2":
                    children[0][8].register_forward_hook(get_activation(layer))
                elif layer == "pool":
                    children[1].register_forward_hook(get_activation(layer))
                elif layer == "output":
                    children[2].register_forward_hook(get_activation(layer))
    return activation
