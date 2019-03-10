import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
import sys, os
from sklearn.externals import joblib
import argparse
from argparse import RawTextHelpFormatter


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

    d = """
    Predicting protein-ligand binding affinities (pKa) with OnionNet model. 
    Citation: Coming soon ... ...
    Author: Liangzhen Zheng
    
    Installation instructions should be refered to 
    https://github.com/zhenglz/onionnet
    
    Examples:
    Show help information
    python predict_pKa.py -h
    
    Predict pKa
    python predict_pKa.py -inp features_300samples.csv -out predicted_pKa_300samples.csv \
                          -scaler Standard_Scaler_OnionNet.model \
                          -model OnionNet_Shell0Missing.h5
    
    """
    parser = argparse.ArgumentParser(description=d, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-inp", type=str, default="features.csv",
                        help="Input. The input file containing the features for the protein-ligand\n"
                             "complexes, one sample per row. This file should be comma separated format\n"
                             "with file header (the first row). ")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. Default is predicted_pKa.csv \n"
                             "The output file name containing the predicted pKa values.")
    parser.add_argument("-scaler", type=str, default="Standard_scaler.model",
                        help="Input. Default is Standard_scaler.model \n"
                              "The standard scaler for feature normalization. \n")
    parser.add_argument("-model", type=str, default="OnionNet.h5",
                        help="Input. Default is OnionNet.h5 \n"
                             "The pre-trained DNN OnionNet model for pKa prediction. \n")

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    # Scale dataset
    if os.path.exists(args.scaler):
        scaler = joblib.load(args.scaler)
    else:
        print("Please put standardscaler model %s in your working directory." % args.scaler)
        sys.exit(0)

    # load the feature dataset
    to_predict = pd.read_csv(args.inp, sep=",", header=0, index_col=0).dropna()

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
    model = tf.keras.models.load_model(args.model,
                                       custom_objects={'RMSE': RMSE,
                                                       'pcc': PCC,
                                                       'PCC': PCC,
                                                       'PCC_RMSE':PCC_RMSE})

    # save output into a file
    y_pred = pd.DataFrame()
    y_pred["ID"] = to_predict.index
    y_pred["pKa_pred"] = model.predict(Xpred).ravel()
    y_pred.to_csv(args.out, sep=",", float_format="%.3f", header=True, index=False)

    if "pKa" in to_predict.columns.values:
        print("PCC : %.3f" % pcc(y_pred['pKa_pred'].values, to_predict.pKa.values))
        print("RMSE: %.3f" % rmse(y_pred['pKa_pred'].values, to_predict.pKa.values))
