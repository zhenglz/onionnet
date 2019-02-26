import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
import sys, os
from sklearn.externals import joblib


def PCC_RMSE(y_true, y_pred):
    """Custom loss function

    Parameters
    ----------
    y_true: np.ndarray, shape = [ N, 1]
        The real pka values
    y_pred: np.ndarray, shape = [ N, 1]
        The predicted pka values

    Returns
    -------
    loss: float
        The customized loss
    """

    alpha = 0.7

    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    rmse = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

    pcc = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)

    return alpha * pcc + (1-alpha) * rmse


def rmse(y_true, y_pred):

    dev = np.square(y_true.ravel() - y_pred.ravel())

    return np.sqrt(np.sum(dev) / y_true.shape[0])


def pcc(y_true, y_pred):
    pcc = stats.pearsonr(y_true, y_pred)
    return pcc[0]


def MAE(y, y_pred):
    dev = y.ravel() - y_pred.ravel()
    ave = np.sum(np.absolute(dev)) / dev.shape[0]

    return ave


def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    return tf.keras.backend.mean(fsp * fst) / (devP * devT)


def remove_shell_features(dat, shell_index, features_n=64):
    """Remove the features in the certein shell

    Parameters
    ----------
    dat: np.ndarray, shape = [ N, M]
        The input dataset
    shell_index: int,
        The shell index.
    features_n: int
        Number of features per shell

    Returns
    -------
    df: np.ndarray
        The output data-frame array
    """
    df = dat.copy()

    start = shell_index * features_n
    end = start + features_n

    zeroes = np.zeros((df.shape[0], features_n))

    df[:, start:end] = zeroes

    return df


if __name__ == "__main__":

    # Scale dataset
    if os.path.exists("StandardScaler_OnionNet.model"):
        scaler = joblib.load("StandardScaler_OnionNet.model")
    else:
        print("Please put standard_scaler.model in your working directory.")
        sys.exit(0)

    # load the feature dataset
    to_predict = pd.read_csv(sys.argv[2], sep=",", header=0, index_col=0).dropna()

    if "success" in to_predict.columns.values:
        to_predict = to_predict[to_predict.success != 0]
        to_predict = to_predict.drop(['success'], axis=1)

    if "pKa" in to_predict.columns.values:
        y_true = to_predict['pKa'].values
        Xpred_scaled = scaler.transform(to_predict.drop(['pKa'], axis=1))
    else:
        Xpred_scaled = scaler.transform(to_predict)

    # set shell 0 features as all-zeroes
    n = 0
    Xpred = remove_shell_features(Xpred_scaled, n, 64).reshape((-1, 64, 60, 1))

    # load the OnionNet Model
    model = tf.keras.models.load_model(sys.argv[1],
                                       custom_objects={'RMSE': RMSE,
                                                       'pcc': PCC,
                                                       'PCC': PCC,
                                                       'PCC_RMSE':PCC_RMSE})

    # save output into a file
    y_pred = pd.DataFrame()
    y_pred["ID"] = to_predict.index
    y_pred["pKa_pred"] = model.predict(Xpred)
    y_pred.to_csv("predicted_"+sys.argv[2], sep=",", float_format="%.3f", header=True, index=False)

