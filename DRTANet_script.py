import os
import argparse
import pickle
import sys
import random
import pandas as pd
from sklearn.svm import SVR
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

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", help="Time series or final value")
parser.add_argument("-v", "--videos", help="Number of videos to use")
args = parser.parse_args()

TYPE = str(args.type)
NB_VIDEOS = args.videos

videos = os.listdir("Videos/")
videos.remove(".DS_Store")

# Load the accumulators (if already computed) (pickles)
if os.path.exists("DRTANet_history_accs_" + str(NB_VIDEOS) + ".p") and os.path.exists(
    "DRTANet_accs_" + str(NB_VIDEOS) + ".p"
):
    with open("DRTANet_history_accs_" + str(NB_VIDEOS) + ".p", "rb") as f:
        detected_changes_history = pickle.load(f)
    with open("DRTANet_accs_" + str(NB_VIDEOS) + ".p", "rb") as f:
        detected_changes = pickle.load(f)
else:
    print("error")
    sys.exit()

# Compute maxium length of history
histories = list(detected_changes_history.values())

MAX_LENGTH = 0
for hist in histories:
    length = len(hist)
    if length > MAX_LENGTH:
        MAX_LENGTH = length
print(MAX_LENGTH)

wd = os.getcwd()
root_dir = wd + "/"
videos_dir = wd + "/Videos/"

# Read excel to load dataset
df = utils.read_excel(root_dir, videos_dir)

# Select videos for testing and the remainder for training
random.seed(42)
test_videos = random.choices(videos, k=8)
train_videos = [x for x in videos if x not in test_videos]

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
    if TYPE == "series":
        accs = detected_changes_history[video]

        # Pad sequence to MAX_LENGTH
        accs = accs + [0] * (MAX_LENGTH - len(accs))
    else:
        accs = detected_changes[video]

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

X = [[el] for el in X_train] if TYPE == "real_duration" else X_train
y = y_train
patients = patients_train
stimuli_names = stimuli_names_train
X_val = [[el] for el in X_test] if TYPE == "real_duration" else X_test
y_val = y_test
patients_val = patients_test
stimuli_names_val = stimuli_names_test

test = pd.DataFrame(
    {
        "x": X,
        "y (s)": y,
    }
)
print(test)

# We'll use a GroupKFold cross validation to ensure that each video is either in the training set and the test set
# The number of folds is the number of videos computed
nb_folds = len(set(stimuli_names))
print("Number of folds : {}".format(nb_folds))

gkf = GroupKFold(n_splits=nb_folds)
regression_model = SVR(kernel="rbf", gamma=0.0001)
grid = {
    "epsilon": [0.01, 0.1, 1, 10],
    "C": [1e1, 1e2, 1e3, 1e4],
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
    verbose=1,
)

print("Grid search.")
grid_search.fit(X, y, groups=stimuli_names)
print("The best parameters are : {}".format(grid_search.best_params_))

# Try to predict for the validation set
y_pred = grid_search.best_estimator_.predict(X_val)

# Get scores obtained for best parameters :
idx = grid_search.cv_results_["params"].index(grid_search.best_params_)

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


with pd.option_context(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.precision",
    2,
    "max_colwidth",
    300,
):
    print(results.head(15))


mse = mean_squared_error(y_val, y_pred)  # MSE
mae = mean_absolute_error(y_val, y_pred)  # MAE
r2 = r2_score(y_val, y_pred)  # R2-SCORE
me = max_error(y_val, y_pred)  # MAX-ERROR
var = explained_variance_score(y_val, y_pred)  # EXPLAINED-VARIANCE

metrics = {
    "mse": mse,
    "mae": mae,
    "r2-score": r2,
    "max-error": me,
    "explained variance score": var,
    "layers": [],
}

df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
print(df_metrics)


folder_name = "exp0bis"


if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save the dictionary with the results
dict_name = folder_name + "/metrics_" + str(TYPE) + ".pkl"

with open(dict_name, "wb") as f:
    pickle.dump(metrics, f)
