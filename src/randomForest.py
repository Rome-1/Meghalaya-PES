import os
from matplotlib import cm
import torch
import numpy as np
import time
import wandb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import sklearn as skl
from torch.utils.data import DataLoader
from models.Training import test_model, train_model
from models.Training import visualize, write_report, ImbalancedDatasetUnderSampler
from preprocessing.Data_maker_loader import with_DSM
import pandas as pd
import matplotlib.pyplot as plt
import pprint
from models.Training import AUC_CM
import sys
from data_loading.get_setup import compute_input_dimensions

region = os.environ.get('REGION')
dir = os.environ.get('DeepForestcast_PATH')

pp = pprint.PrettyPrinter(indent=4)

hyperparameter_defaults = dict(
    region=region,
    size=7,
    train_times=1,
    test_times=1,
    # Set training time period
    start_year=18,
    end_year=22,
    modeltype="3D",
    years_ahead=1,
    w=10
)

data_layers = {
        "static": {
        # /outputs/{region}/external/ + "static_layer_name": True/False,
    },
    "time": {
        # /outputs/{region}/tensors/ + "static_layer_name": True/False,

    },
}

config = hyperparameter_defaults
print("RANDOM FOREST")
pp.pprint(config)
# Years

perc = (100 * config["train_times"]) / (
    config["train_times"] + 1
)  # the percentile to for threshold selection. Advisable to be 100*times/(times+1)



# WHERE TO IMPORT DATA FROM
wherepath = dir + "/outputs/" + config["region"] + "/tensors"
savepath = dir + "/models/" + config["region"] + "_models/random_forest"
if not os.path.exists(savepath):
    os.makedirs(savepath)

# WHERE TO SAVE MODEL CHECKPOINT
modelpath = dir + "/models/" + config["region"] + "_models/3D"
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

# WHERE TO SAVE IMAGES TRACKING TRAINING PROCESS
picspath = dir + "/models/" + config["region"] + "_models/3D/pics"
if not os.path.exists(picspath):
    os.makedirs(picspath)

# WHERE TO SAVE MODEL PERFORMANCE OF EACH JOB FOR TRAIN, VAL AND TEST DATA
file = (
    dir
    + "/models/"
    + config["region"]
    + "_models/3D/grid_summary/ConvRNN.Conv_3D.txt"
)
if not os.path.exists(os.path.dirname(file)):
    os.makedirs(os.path.dirname(file))

input_dim_2D, input_dim_3D = compute_input_dimensions(data_layers)

DSM = False
Data = with_DSM(
    size=int(config["size"] / 2),
    start_year=config["start_year"]-1,
    end_year=config["end_year"]-1,
    wherepath=wherepath,
    DSM=DSM,
    data_layers=data_layers,
    years_ahead=config["years_ahead"],
    type=config["modeltype"],
)
# Data[0][0].flatten().shape

train_idx = np.load(wherepath + "/" + "Train3D_idx%d.npy" % (config["end_year"]-1))
print("Length train ids:", len(train_idx))
test_idx = np.load(wherepath + "/" + "Test3D_idx%d.npy" % (config["end_year"]-1))
print("Length same year test ids:", len(test_idx))

def format_loader(loader):
    for batch, (data, target, cor) in enumerate(loader, 1):
        if type(data) == type([]):
            data_0 = data[0].reshape(data[0].shape[0], -1) # flatten features
            data_1 = data[1].reshape(data[1].shape[0], -1) # flatten features
            data_combine = torch.cat([data_0, data_1], dim=1) # combine along features
            print("data combine (list):", data_combine.shape)
            
        else:
            dshape = data.shape
            data_combine = data.reshape([dshape[0], dshape[1] * dshape[2] * dshape[3]]) # flatten features
            print("data combine (val):", data_combine.shape)

    dataflat = data_combine.numpy()
    return dataflat, target, cor

# Set train and test samplers
train_sampler = ImbalancedDatasetUnderSampler(
    labels=Data.labels, indices=train_idx, times=config["train_times"]
)
tr_idx = train_sampler.classIndexes_unsampled
test_sampler = ImbalancedDatasetUnderSampler(
    labels=Data.labels, indices=test_idx, times=config["test_times"]
)

batch_size = tr_idx.size
print("Train sample (which is batch) size:", batch_size)
train_loader = DataLoader(
    Data, sampler=train_sampler, batch_size=batch_size, drop_last=True
)

dataflat, target, cor = format_loader(train_loader)


# Parameter Space ————————————————————————————
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=200, stop=dataflat.shape[1], num=5)] # greater number of estimators can actually reduce overfit
# # Number of features to consider at every split
# max_features = ["log2", "sqrt"] # to reduce overfitting # int in the range [1, inf), a float in the range (0.0, 1.0], a str among {'log2', 'sqrt'} or None
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 40, num=11)]
# # max_depth.append(None) # to reduce overfitting
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [False] # to reduce overfitting
# # Create the random grid
# random_grid = {
#     "n_estimators": n_estimators,
#     "max_features": max_features,
#     "max_depth": max_depth,
#     "min_samples_split": min_samples_split,
#     "min_samples_leaf": min_samples_leaf,
#     "bootstrap": bootstrap,
# }
# pp.pprint(random_grid)
# END: Parameter Space ————————————————————————————

params = {   'bootstrap': False,
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 5,
    'min_samples_split': 5,
    'n_estimators': 100, # int(sys.argv[1]) # to prevent overfitting
    } 
pp.pprint(params)

clf = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1, **params) # verbose=2
# rf = clf = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1) # verbose=2
# clf = RandomizedSearchCV(
#    estimator=rf,
#    param_distributions=random_grid,
#    n_iter=20,
#    cv=3,
#    verbose=2,
#    random_state=42,
#    n_jobs=-1,
# )

print("Fitting...")
dataflat = dataflat.copy()
clf.fit(dataflat, target)

# Paper
# clf.best_params_
# {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': False}
# Us (search space limited for overfitting)
# {   'bootstrap': False,
#     'max_depth': 37,
#     'max_features': 'sqrt',
#     'min_samples_leaf': 1,
#     'min_samples_split': 2,
#     'n_estimators': 850}
# pp.pprint(clf.best_params_)

clf.score(dataflat, target)

predtrain = clf.predict(dataflat)
cm = confusion_matrix(target, predtrain)
print(cm)
cr = classification_report(target, predtrain)
print(cr)


def predict_print_plot_grouped(dataflat, target, label, start_year, end_year):
    print("=================================")
    dataflat = dataflat.copy()
    print(label, str(end_year))
    print("Predicting...")
    pred = clf.predict(dataflat)

    cm = confusion_matrix(target, pred)
    cr = classification_report(target, pred)
    print(cr)

    # AUC, CM, Metrics Calculation (without using threshold because predications are binary)
    matrix = cm # skl.metrics.confusion_matrix(target, pred)
    w = config["w"]
    cost = (matrix[0, 1] + w * matrix[1, 0]) / (np.sum(matrix) + (w - 1) * matrix[1, 0])
    auc = skl.metrics.roc_auc_score(target, pred)
    print("\tAUC :", auc)
    print("\tPred:  ", "\t", 0, "\t\t", 1)
    print("\tTrue: 0", "\t", matrix[0, 0], "\t\t", matrix[0, 1])
    print("\tTrue: 1", "\t", matrix[1, 0], "\t\t", matrix[1, 1])
    print("\tAccuracy ", np.trace(matrix) / np.sum(matrix))
    print("\tTrue Positive Rate ", matrix[1, 1] / sum(matrix[1, :]))
    print("\tCost = FP+w*FN/TP+FP+w*FN+TN = %.4f (w = %d)" % (cost, w))

    # Extract feature importance
    importances = clf.feature_importances_
    std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)

    # Prep for chunking (CNN window) feature importance
    chunk_size = config["size"] ** 2
    n_features = len(importances)
    static_group, temporal_group = input_dim_3D
    temporal_group *= (end_year - start_year) # total temporal layers

    # Indices for chunking
    first_part_end = static_group * chunk_size
    second_part_end = (static_group + temporal_group) * chunk_size
    assert n_features >= second_part_end, "Not enough features to split into the required chunks."

    # Aggregate feature importances and standard deviations for the first static and temporal chunk groups
    grouped_importances_first = [importances[i:i + chunk_size].sum() for i in range(0, first_part_end, chunk_size)]
    grouped_std_first = [std[i:i + chunk_size].sum() for i in range(0, first_part_end, chunk_size)]
    grouped_importances_second = [importances[i:i + chunk_size].sum() for i in range(first_part_end, second_part_end, chunk_size)]
    grouped_std_second = [std[i:i + chunk_size].sum() for i in range(first_part_end, second_part_end, chunk_size)]

    grouped_importances_first_series = pd.Series(grouped_importances_first)
    grouped_std_first_series = pd.Series(grouped_std_first)
    grouped_importances_second_series = pd.Series(grouped_importances_second)
    grouped_std_second_series = pd.Series(grouped_std_second)

    # Plot for the static chunks (20) 
    fig, ax = plt.subplots()
    grouped_importances_first_series.plot.bar(yerr=grouped_std_first_series, ax=ax)
    ax.set_title(f"Grouped Feature Importances (Static {chunk_size}-feature chunks)")
    ax.set_ylabel("Sum of Mean Decrease in Impurity")
    ax.set_xlabel(f"Feature Group (CNN window of size {chunk_size})")
    fig.tight_layout()
    fig.show()
    savefig_path = savepath + "/" + f"feature_importance_static_{label}_" + str(end_year) + ".png"
    fig.savefig(savefig_path)
    print("Saved fig:", savefig_path)

    # Plot for the temporal chunks (40)
    fig, ax = plt.subplots()
    grouped_importances_second_series.plot.bar(yerr=grouped_std_second_series, ax=ax)
    ax.set_title(f"Grouped Feature Importances (Temporal {chunk_size}-feature chunks)")
    ax.set_ylabel("Sum of Mean Decrease in Impurity")
    ax.set_xlabel(f"Feature Group (CNN window of size {chunk_size})")
    fig.tight_layout()
    fig.show()
    savefig_path = savepath + "/" + f"feature_importance_temporal_{label}_" + str(end_year) + ".png"
    fig.savefig(savefig_path)
    print("Saved fig:", savefig_path)

    # Create lists to store chunk results
    chunk_numbers = []
    grouped_importances = []
    grouped_std = []
    grouped_max = []

    # Aggregate feature importances and standard deviations for all chunks
    for i in range(0, second_part_end, chunk_size):
        chunk_numbers.append(f"Chunk {i//chunk_size + 1}")
        grouped_importances.append(importances[i:i + chunk_size].sum())
        grouped_std.append(std[i:i + chunk_size].sum())
        grouped_max.append(max(importances[i:i + chunk_size]))

    # Create a DataFrame to display the results
    table_data = pd.DataFrame({
        'Feature Chunk': chunk_numbers,
        'Sum of Feature Importances': grouped_importances,
        'Sum of Standard Deviations': grouped_std
    })

    print(table_data)



# TESTING WITHIN YEAR ——————————————————————————————————
test_loader = DataLoader(
    Data, sampler=test_sampler, batch_size=test_idx.size, drop_last=False
)
dataflattest, targettest, cortest = format_loader(test_loader)
predict_print_plot_grouped(dataflattest, targettest, "same_year_test", config["start_year"]-1, config["end_year"]-1)

# END: TESTING WITHIN YEAR ——————————————————————————————————



# TESTING NEXT YEAR ——————————————————————————————————

# Test On Next Year [start_year, end_year] predicting end_year+1
start_year = config["start_year"]
end_year = config["end_year"]

Data = with_DSM(
    size=int(config["size"] / 2),
    start_year=start_year,
    end_year=end_year,
    wherepath=wherepath,
    DSM=DSM,
    data_layers=data_layers,
    years_ahead=config["years_ahead"],
    type=config["modeltype"],
)

test_idx = np.load(wherepath + "/" + "Test3D_idx%d.npy" % (config["end_year"]))
print("Length next-year samples:", len(test_idx))
test_sampler = ImbalancedDatasetUnderSampler(
    labels=Data.labels, indices=test_idx, times=config["test_times"]
)
test_loader = DataLoader(
    Data, sampler=test_sampler, batch_size=test_idx.size, drop_last=False
)
dataflattest, targettest, cortest = format_loader(test_loader)

predict_print_plot_grouped(dataflattest, targettest, "next_year_test", start_year, end_year)

# END: TESTING NEXT YEAR ——————————————————————————————————