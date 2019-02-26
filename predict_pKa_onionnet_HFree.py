import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from scipy import stats
import itertools
from sklearn.externals import joblib

import os, sys


def RMSE(y, y_pred):
    dev = y - y_pred
    dev = dev ** 2

    return np.sqrt(np.sum(dev) / dev.shape[0])


def PCC(y, y_pred):
    pcc = stats.pearsonr(y, y_pred)
    return pcc


def SD(y, y_pred):

    dev = y - y_pred
    dev = dev ** 2

    return np.sqrt(np.sum(dev) / (dev.shape[0] - 1))


def MAE(y, y_pred):
    dev = y - y_pred
    ave = np.sum(np.absolute(dev)) / dev.shape[0]

    return ave


def remove_features_HFree(df, shell_index=0):
    original_features = df.columns.values

    remove_ndx = range(shell_index * 64, shell_index * 64 + 64)
    remove_features = [str(x) for x in remove_ndx]

    keep_features = [x for x in original_features if (x.split("_")[2] not in remove_features)]

    kf = [x for x in keep_features if (x.split("_")[0] != "H" and x.split("_")[1] != "H")]
    return kf


# Scale dataset
if os.path.exists("standard_scaler.model"):
    scaler = joblib.load("standard_scaler.model")
else:
    print("Please put standard_scaler.model in your working directory.")
    sys.exit(0)

input_shape = (-1, 49, 59, 1)

# load dataset
X_topred = pd.read_csv(sys.argv[2], sep=",", header=0, index_col=0)
kf = remove_features_HFree(X_topred, 0)

# scale dataset
Xscale = scaler.transform(X_topred[kf]).reshape(input_shape)

# load model
model = tf.keras.models.load_model(sys.argv[1])

# predict pKa
ypred = model.predict(Xscale)
ypred = np.ravel(ypred)

# save prediction to a file
df = pd.DataFrame()
df["PDBID"] = X_topred.index.values
df["pKa_predict"] = ypred

df.to_csv("predicted_HFree_" + sys.argv[2], sep=",", header=True, index=True, float_format="%.3f")

