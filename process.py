import cv2
import copy
import torch
import pickle
import numpy as np

import utils


def compute_threshold(prev_thresh, last_time, layer, timestep, params):
    """
    This function computes the new value of the threshold.

    Args:
      prev_thresh : previous value of the threshold
      last_time : last time the threshold was updated
      layer : layer number
      timestep : current time step

    Returns:
      thresh : new value of the threshold
    """

    # Parameters for implementing salient event threshold

    """
    T_max : maximum threshold value for kth klayer
    T_min : minimum threshold value for kth klayer
    tau : decay time constant for kth klayer
    alpha : dividing constant to adjust the variance of the noise
    """
    # print(params[layer]["T_max"], params[layer]["T_min"], layer)
    std = (params[layer]["T_max"] - params[layer]["T_min"]) / params[layer]["alpha"]
    # sample normal distribution of mean 0, std
    rand = np.random.normal(0, std)
    # Compute the number of elapsed time steps since the last update
    D = timestep - last_time[layer]
    thresh = (
        prev_thresh
        - ((params[layer]["T_max"] - params[layer]["T_min"]) / params[layer]["tau"])
        * np.exp(-(D / (params[layer]["tau"])))
        + rand
    )

    return thresh


def reset_threshold(layer, params) -> float:
    """
    This function resets the threshold to its maximum value.

    Args:
      layer : layer number

    Returns:
      thresh : new value of the threshold
    """

    # Parameters for implementing salient event threshold

    """
    T_max : maximum threshold value for kth klayer
    T_min : minimum threshold value for kth klayer
    tau : decay time constant for kth klayer
    alpha : dividing constant to adjust the variance of the noise
  """
    return params[layer]["T_max"]


def process_video(
    model,
    device,
    activation,
    video_path,
    network,
    layers,
    params,
    display=False,
    frequency=30,
):
    """
    This function processes a video and computes accumulators as well as the sense of time and change detection plots.

    Args:
        model : pretrained model
        device : device on which the model is loaded
        activation : activation of the model
        video_path : path to the video
        network : network architecture
        layers : layers of the network to be considered
        params : parameters for implementing the accumulation mechanism
        display : whether to display the video or not (default = False)
        frequency : frequency of the frames to be processed (default = 30)

    Returns:
        accs : accumulators
        accumulators_history : history of the accumulators over time
        thresholds : values of the thresholds over time
        l2 : L2 norms between states of the activation over time
    """

    l2 = {
        layer: [] for layer in layers
    }  # dict that holds the L2 norms between states of the activation
    accs = {layer: 0 for layer in layers}  # accumulators
    accumulators_history = {
        layer: [] for layer in layers
    }  # history of the accumulators

    # For each layer, register the last timestep the threshold was reset
    last_time = {layer: 0 for layer in layers}

    # For each layer in the network, initialize an empty list that will contain
    # the values of the threshold over time
    thresholds = {layer: [] for layer in layers}

    # Reading video file
    vid = cv2.VideoCapture(video_path)
    fps = vid.get(cv2.CAP_PROP_FPS)

    # Loop over the frames of the video
    timestep = 0

    # For each layer in the network (conv2, pool5, fc7, output) initialize
    # a tensor of zeros with the same size as the output of the layer
    # by doing a forward pass with the first frame of the video
    frame = vid.read()[1]
    # Transform
    frame = utils.transform(network, frame)

    # Add a batch dimension
    batch = torch.unsqueeze(frame, 0)

    # Move to device
    batch = batch.to(device)
    model.to(device)
    output = model(batch)
    prevs = {layer: torch.zeros(activation[layer].shape) for layer in layers}

    while True:
        # grab current frame
        _, frame = vid.read()

        if frame is None:
            break

        # Display frame
        if display:
            cv2.imshow("frame", frame)
            cv2.waitKey()
            cv2.destroyAllWindows()

        # Press Q on keyboard to exit
        if cv2.waitKey(1) == ord("q"):
            break

        if timestep % frequency == 0:
            # Apply the transform to the frame depending on the network
            processed_frame = utils.transform(network, frame)

            # Form a batch (only one image)
            batch = torch.unsqueeze(processed_frame, 0)

            batch = batch.to(device)
            model.to(device)

            # Pass frame into network and get the output of the model
            with torch.no_grad():
                output = model(batch)

            # Here compute the L2 norm between current activations and the previous ones
            for layer in layers:
                norm = np.linalg.norm(
                    activation[layer].flatten().cpu() - prevs[layer].flatten().cpu()
                )
                l2[layer].append(norm)
                # The threshold decays with some stochasticity over time and has an initial value
                # for each layer
                if timestep == 0:
                    initial_threshold = reset_threshold(layer, params)
                    thresh = compute_threshold(
                        initial_threshold, last_time, layer, timestep, params
                    )
                else:
                    thresh = compute_threshold(
                        thresholds[layer][-1], last_time, layer, timestep, params
                    )

                # Reset feature map accumulator if the L2 norm is above the threshold
                if norm >= thresh:
                    accs[layer] += 1
                    thresh = reset_threshold(layer, params)
                    last_time[layer] = timestep

                thresholds[layer].append(thresh)
                accumulators_history[layer].append(accs[layer])

            prevs = copy.deepcopy(activation)

        timestep = timestep + 1

    if display:
        vid.release()
        cv2.destroyAllWindows()
        utils.show_classification(output)

    return accs, thresholds, l2, accumulators_history


def compute_accumulators(
    model,
    network,
    layers,
    params,
    activation,
    device,
    videos_dir,
    number_of_videos,
    plot,
    results_dir,
):
    """
    This function computes the accumulators for a set of videos.

    Args:
        model : pretrained model
        network : network architecture
        layers : layers of the network to be considered
        params : parameters for implementing the accumulation mechanism
        activation : activation of the model
        device : device on which the model is loaded
        videos_dir : directory where the videos are stored
        number_of_videos : number of videos to be processed
        plot : whether to plot the change detection and sense of time plots or not
        results_dir : directory where the results are stored

    Returns:
        videos_accs : accumulators for each video
        videos_thresholds : values of the thresholds over time for each video
        videos_l2 : L2 norms between states of the activation over time for each video
    """
    videos = utils.retrieve_videos(
        videos_dir, number_of_videos=number_of_videos, video_format=".avi"
    )
    frequency = 30

    videos_accs = {}
    videos_thresholds = {}
    videos_l2 = {}
    videos_accs_histories = {}

    for video in videos:
        video_path = videos_dir + video
        accs, thresholds, l2, accumulators_history = process_video(
            model,
            device,
            activation,
            video_path,
            network,
            layers,
            params,
            display=False,
            frequency=frequency,
        )
        # print(accumulators_history)
        if plot:
            utils.plot_change_detection(
                layers, l2, thresholds, params, results_dir, video
            )
            utils.plot_sense_of_time(layers, accumulators_history, results_dir, video)

        videos_accs[video] = accs
        videos_thresholds[video] = thresholds
        videos_l2[video] = l2
        videos_accs_histories[video] = accumulators_history

    print("Processed {} videos".format(number_of_videos))

    # Save the dictionaries as pickles
    pickle.dump(
        videos_accs,
        open("videos_accs_" + str(number_of_videos) + "_" + str(network) + ".p", "wb"),
    )
    pickle.dump(
        videos_thresholds,
        open(
            "videos_thresholds_" + str(number_of_videos) + "_" + str(network) + ".p",
            "wb",
        ),
    )
    pickle.dump(
        videos_l2,
        open("videos_l2_" + str(number_of_videos) + "_" + str(network) + ".p", "wb"),
    )
    pickle.dump(
        videos_accs_histories,
        open(
            "videos_accs_histories_"
            + str(number_of_videos)
            + "_"
            + str(network)
            + ".p",
            "wb",
        ),
    )

    return videos_accs, videos_thresholds, videos_l2
