import os
import sys
import argparse
import pickle
import random
import pandas as pd

from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error, explained_variance_score
import utils

#################################
# Define parameters
#################################
wd = os.getcwd()
videos_dir = wd + "/Videos/"
root_dir = wd + "/"

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--videos", help="Number of videos to use", default=10)
parser.add_argument("-t", "--type", help="Baseline type (dummy or real duration)")
args = parser.parse_args()

NB_VIDEOS = str(args.videos)
TYPE = str(args.type)

print("Number of videos: " + str(NB_VIDEOS))

df = utils.read_excel(root_dir, videos_dir)

#################################
# Load dataset
#################################
# Load the accumulators (if already computed) (pickles)
NETWORK = "alexnet" # not important, preds are not based on network
if (
    os.path.exists("videos_accs_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p")
    and os.path.exists("videos_thresholds_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p")
    and os.path.exists("videos_l2_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p")
):
    with open("videos_accs_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p", "rb") as f:
        videos_accs = pickle.load(f)
    with open("videos_thresholds_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p", "rb") as f:
        videos_thresholds = pickle.load(f)
    with open("videos_l2_" + str(NB_VIDEOS) + "_" + str(NETWORK) + ".p", "rb") as f:
        videos_l2 = pickle.load(f)
else:
    print("Error: Accumulators not found, please compute accumulators first.")
    sys.exit()


# Constitue training set and validation set (based on Excel for y_true)
if TYPE == "dummy":
    X, y, _, _, X_val, y_val, _, _ = utils.build_dataset(videos_dir, videos_accs, df)
elif TYPE == "real_duration":
    X, y, _, _, X_val, y_val, _, _ = utils.build_duration_dataset(videos_dir, df)
print("Dataset constituted.")


#################################
# Compute baseline
#################################
if TYPE == "dummy":
    print("--------------------BASELINE : DUMMY-------------------")
    model = DummyRegressor(strategy='mean')
elif TYPE == "real_duration":
    print("----------------BASELINE : REAL DURATIONS---------------")
    model = SVR(kernel="rbf", C=1e3, gamma=0.0001, epsilon=0.1)
model.fit(X, y)

#################################
# Test
#################################
y_pred = model.predict(X_val)
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

mse = mean_squared_error(y_val, y_pred) # MSE
mae = mean_absolute_error(y_val, y_pred) # MAE
r2 = r2_score(y_val, y_pred) # R2-SCORE
me = max_error(y_val, y_pred) # MAX-ERROR
var = explained_variance_score(y_val, y_pred) # EXPLAINED-VARIANCE 

metrics = {"mse": mse, "mae": mae, "r2-score": r2, "max-error": me, "explained variance score": var, "layers": []}
print(metrics)

#################################
# Save
#################################
folder_name = "exp0"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save the dictionary with the results
if TYPE == "dummy":
    dict_name = folder_name + "/B0_dummy_metrics.pkl"
elif TYPE == "real_duration":
    dict_name = folder_name + "/B0_real_duration_metrics.pkl"

with open(dict_name, 'wb') as f:
    pickle.dump(metrics, f)