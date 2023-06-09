import os
import sys
import argparse
import torch

import regression

"""
    This script is used to automate the process described in the paper.
"""
if __name__ == "__main__":
    #################################
    # Define device
    #################################
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print("Using device: " + device)

    #################################
    # User input
    #################################
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model to use")
    parser.add_argument("-v", "--videos", help="Number of videos to use", default=10)
    parser.add_argument("-t", "--type", help="Type of run : vanilla - naive - lasso")
    parser.add_argument(
        "-f", "--features", help="Maximum number of features to use", default=None
    )
    parser.add_argument("-id", "--id", help="Job ID")

    # Plot
    parser.add_argument("--p", action="store_true")
    parser.add_argument("--no-p", dest="p", action="store_false")
    parser.set_defaults(p=False)
    # Calculate accumulators
    parser.add_argument("--c", action="store_true")
    parser.add_argument("--no-c", dest="c", action="store_false")
    parser.set_defaults(c=False)

    args = parser.parse_args()

    network = str(args.model)
    nb_videos = int(args.videos)
    type = str(args.type)
    features = int(args.features) if args.features is not None else None
    jobid = int(args.id) if args.id is not None else None

    p = args.p
    c = args.c

    # Parameters
    print("Using model: " + network)
    print("Number of videos: " + str(nb_videos))
    print("Execution type : " + str(type))
    if type == "naive":
        print("Using " + str(features) + " features")
        print("Job ID : " + str(jobid))

    # Options
    print("Plotting ? ", "yes" if p == True else "no")
    print("Calculate accumulators ? ", "yes" if c == True else "no")

    #################################
    # Directory paths
    #################################
    wd = os.getcwd()
    root_dir = wd + "/"
    videos_dir = wd + "/Videos/"

    # Check if the directory exists and report error if not
    if not os.path.exists(videos_dir):
        print("Error: Videos directory does not exist")
        sys.exit()
    else:
        print(
            "The directory contains : " + str(len(os.listdir(videos_dir))) + " videos"
        )

    # Create new directory to store results
    new_dir = wd + "/results/" + str(network) + "_" + str(nb_videos) + "/"

    try:
        os.mkdir(new_dir)
    except FileExistsError:
        pass

    #################################
    # Run main process
    #################################
    regression.compute(
        device,
        network,
        nb_videos,
        type,
        features,
        jobid,
        p,
        c,
        root_dir,
        videos_dir,
        new_dir,
    )
