import os
import sys
import pickle
import itertools
import random
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    max_error,
    explained_variance_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GridSearchCV,
)

import utils
import process


def compute(
    DEVICE,
    NETWORK,
    NB_VIDEOS,
    TYPE,
    FEATURES,
    JOBID,
    p,
    c,
    root_dir,
    videos_dir,
    new_dir,
):
    """
    This function implements the complete pipeline of the regression task, including:
        - Model architecture definition
        - Accumulators computation for selected videos and layers
        - Vanilla regression scheme, LassoCV regression scheme, or Naive regression scheme
        - Results saving

    Args:
        DEVICE (str): Device to use for the computation (mps, cpu or cuda)
        NETWORK (str): Network to use for the computation (alexnet, resnet18 or efficientnetB0, efficientnetB4 or efficientnetB7)
        NB_VIDEOS (int): Number of videos to use for the computation
        TYPE (str): Type of regression scheme to use (vanilla, lasso or naive)
        FEATURES (int): Number of features to use for the computation of naive regression scheme (combinations of #features layers)
        JOBID (int): Job ID for the computation of the naive regression scheme
        p (bool): Boolean to indicate if the plots should be displayed
        c (bool): Boolean to indicate if the accumulators should be recomputed
        root_dir (str): Root directory of the project
        videos_dir (str): Directory of the videos
        new_dir (str): Directory to save the results

    Returns:
        None
    """
    #################################
    # Define model architecture
    #################################
    MODEL = utils.get_model(NETWORK)
    MODEL.eval()  # move to evaluation mode for testing

    # Read excel to load dataset
    df = utils.read_excel(root_dir, videos_dir)

    #################################
    # Compute accumulators on all layers
    #################################

    # Computes the accumulators videos in directory.
    if c:  # To recompute the accumulators on a set of videos
        # Retrieves all the layers of the NETWORK
        LAYERS = utils.get_layers(NETWORK, None)

        # Global variables
        PARAMS = utils.get_params(None, NETWORK, LAYERS)

        # Hook all layers of the network
        ACTIVATION = utils.hook_layers(MODEL, NETWORK, None, LAYERS)

        videos_accs, videos_thresholds, videos_l2 = process.compute_accumulators(
            MODEL,
            NETWORK,
            LAYERS,
            PARAMS,
            ACTIVATION,
            DEVICE,
            videos_dir,
            NB_VIDEOS,
            p,
            new_dir,
        )

    else:
        # Load the accumulators (if already computed) (pickles)
        if (
            os.path.exists("videos_accs_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p")
            and os.path.exists(
                "videos_thresholds_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p"
            )
            and os.path.exists(
                "videos_l2_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p"
            )
        ):
            with open(
                "videos_accs_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p", "rb"
            ) as f:
                videos_accs = pickle.load(f)
            with open(
                "videos_thresholds_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p", "rb"
            ) as f:
                videos_thresholds = pickle.load(f)
            with open(
                "videos_l2_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p", "rb"
            ) as f:
                videos_l2 = pickle.load(f)
        else:
            print("Error: Accumulators not found, please compute accumulators first.")
            sys.exit()

    #################################
    # Generate combinations of layers
    #################################

    # Retrieves all the layers of the NETWORK
    LAYERS = utils.get_layers(NETWORK, TYPE)
    print(LAYERS)

    # Global variables
    PARAMS = utils.get_params(TYPE, NETWORK, LAYERS)
    print(PARAMS)

    if TYPE == "lasso" or TYPE == "vanilla":
        combines = [
            LAYERS
        ]  # Get all layers of the model (or sleected layers for vanilla)
    elif TYPE == "naive":
        combines = []
        for f in range(1, FEATURES + 1):
            # Generate all the possible combinations of $f layers
            temp = list(itertools.combinations(LAYERS, f))
            # generated as a list of tuples -> convert to list of lists
            for i in range(len(temp)):
                combines.append(list(temp[i]))

    #################################
    # Iterate over all the combinations
    #################################
    print("Iterating over following combinations of layers : ", combines)
    if TYPE == "naive" and JOBID is not None:
        combines = [combines[JOBID]]
        print(combines)
    for combine in combines:
        # Reduce the accumulators to those of selected layers only
        LAYERS = combine
        accs = {}
        for v in list(videos_accs.keys()):
            all_accs = videos_accs[v]  # get accumulators video v for all layers
            accs[v] = dict((k, all_accs[k]) for k in LAYERS if k in all_accs)

        # Constitue training set and validation set (based on Excel)
        (
            X,
            y,
            patients,
            stimuli_names,
            X_val,
            y_val,
            patients_val,
            stimuli_names_val,
        ) = utils.build_dataset(videos_dir, accs, df)
        print("Dataset constituted.")

        ##################################
        # Vanilla
        ##################################
        if TYPE == "vanilla":
            print("--------------------VANILLA IMPLEMENTATION----------------------")
            # Train and predict
            regression_model = SVR(kernel="rbf", C=0.001, gamma=0.0001, epsilon=0.1)

            regression_model.fit(X, y)
            y_pred = regression_model.predict(X_val)

            metric_scores = None
            model_params = regression_model.get_params()

        ##################################
        # LassoCV
        ##################################
        elif TYPE == "lasso":
            print("--------------------LASSO-CV---------------------")

            feature_names = LAYERS

            # We'll use a GroupKFold cross validation to ensure that each video is either in the training set and the test set
            # The number of folds is the number of videos computed
            nb_folds = len(set(stimuli_names))
            print("Number of folds : {}".format(nb_folds))
            gkf = GroupKFold(n_splits=nb_folds)
            splits = gkf.split(X, groups=stimuli_names)  # splits generator

            lassoCV = LassoCV(cv=splits)
            lasso = lassoCV.fit(X, y)

            importance = np.abs(lasso.coef_)
            y_pred = lassoCV.predict(X_val)

            # Get scores obtained for best parameters :
            idx = list(lassoCV.alphas_).index(lassoCV.alpha_)

            # Compute the uncertainty over folds
            # Get all splits scores
            metric_scores = lassoCV.mse_path_[idx]

            print("Metrics obtained during CV : ")
            print(metric_scores)

            model_params = lasso.get_params()
            model_params = {
                el: model_params[el]
                for el in model_params
                if el in ["alphas", "eps", "max_iter", "n_alphas", "tol"]
            }
            print(model_params)

        ##################################
        # GridSearchCV
        ##################################
        elif TYPE == "naive":
            print("--------------------NAIVE IMPLEMENTATION-------------------")

            # We'll use a GroupKFold cross validation to ensure that each video is either in the training set and the test set
            # The number of folds is the number of videos computed
            nb_folds = len(set(stimuli_names))
            print("Number of folds : {}".format(nb_folds))

            gkf = GroupKFold(n_splits=nb_folds)
            regression_model = SVR(epsilon=0.1, kernel="rbf")
            grid = {
                "C": [1, 1e1, 1e2, 1e3, 1e4, 1e5],
            }

            metrics = [
                "r2",
                "neg_mean_absolute_error",
                "neg_mean_squared_error",
                "max_error",
                "explained_variance",
            ]
            grid_search = GridSearchCV(
                estimator=regression_model,
                param_grid=grid,
                cv=gkf,
                scoring=metrics,
                refit="neg_mean_squared_error",
            )

            print("Grid search.")
            grid_search.fit(X, y, groups=stimuli_names)
            print("The best parameters are : {}".format(grid_search.best_params_))

            # Try to predict for the validation set
            y_pred = grid_search.best_estimator_.predict(X_val)

            # Get scores obtained for best parameters :
            idx = grid_search.cv_results_["params"].index(grid_search.best_params_)

            # Compute the uncertainty over folds
            # Get all splits scores
            metric_scores = {key: [] for key in metrics}
            for metric in metrics:
                scores = []
                for fold in range(nb_folds):
                    name = "split" + str(fold) + "_test_" + metric
                    res = list(grid_search.cv_results_[name])
                    scores.append(res)

                metric_scores[metric] = np.array(scores)[:, idx]

            print("Metrics obtained during CV : ")
            print(metric_scores)

            model_params = grid_search.best_params_

        # Display results in dataframe
        results = pd.DataFrame(
            {
                "accs": X_val,
                "y_pred (s)": y_pred,
                "y_true (s)": y_val,
            }
        )
        # Shuffle the rows of the DataFrame
        results = results.iloc[random.sample(range(len(results)), len(results))]
        print(results.head(15))

        mse = mean_squared_error(y_val, y_pred)  # MSE
        mae = mean_absolute_error(y_val, y_pred)  # MAE
        r2 = r2_score(y_val, y_pred)  # R2-SCORE
        me = max_error(y_val, y_pred)  # MAX-ERROR
        var = explained_variance_score(y_val, y_pred)  # EXPLAINED-VARIANCE

        # Print
        if TYPE == "lasso":
            metrics = {
                "mse": mse,
                "mae": mae,
                "r2-score": r2,
                "max-error": me,
                "explained variance score": var,
                "layers": list(np.array(feature_names)[importance > 0]),
            }
        else:
            metrics = {
                "mse": mse,
                "mae": mae,
                "r2-score": r2,
                "max-error": me,
                "explained variance score": var,
                "layers": LAYERS,
            }

        print(metrics)

        # Save everything
        if TYPE == "naive":
            if NETWORK == "alexnet":
                folder_name = "exp3"
            elif NETWORK == "resnet18":
                folder_name = "exp5"
            elif NETWORK == "efficientnetB0":
                folder_name = "exp7"
            elif NETWORK == "efficientnetB4":
                folder_name = "exp9"
            elif NETWORK == "efficientnetB7":
                folder_name = "exp11"

        elif TYPE == "lasso":
            if NETWORK == "alexnet":
                folder_name = "exp2"
            elif NETWORK == "resnet18":
                folder_name = "exp4"
            elif NETWORK == "efficientnetB0":
                folder_name = "exp6"
            elif NETWORK == "efficientnetB4":
                folder_name = "exp8"
            elif NETWORK == "efficientnetB7":
                folder_name = "exp10"

        elif TYPE == "vanilla":
            if NETWORK == "alexnet":
                folder_name = "exp1"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Save the dictionary with the results
        if TYPE == "naive":
            if metric_scores is not None:
                uncertainty_name = (
                    folder_name + "/uncertainty_metrics_" + str(JOBID) + ".pkl"
                )
            dict_name = folder_name + "/metrics_" + str(JOBID) + ".pkl"
            params_name = folder_name + "/params_" + str(JOBID) + ".pkl"
        else:
            if metric_scores is not None:
                uncertainty_name = folder_name + "/uncertainty_metrics.pkl"
            dict_name = folder_name + "/metrics.pkl"
            params_name = folder_name + "/params.pkl"

        with open(dict_name, "wb") as f:
            pickle.dump(metrics, f)

        with open(params_name, "wb") as f:
            pickle.dump(model_params, f)

        if metric_scores is not None:
            with open(uncertainty_name, "wb") as f:
                pickle.dump(metric_scores, f)
