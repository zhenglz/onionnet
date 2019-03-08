import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.externals import joblib
import argparse
from argparse import RawTextHelpFormatter
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
                        helpp="Input. Default is Standard_scaler.model \n"
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

    input_shape = (-1, 49, 59, 1)

    # load dataset
    X_topred = pd.read_csv(args.inp, sep=",", header=0, index_col=0)
    kf = remove_features_HFree(X_topred, 0)

    # scale dataset
    Xscale = scaler.transform(X_topred[kf]).reshape(input_shape)

    # load model
    model = tf.keras.models.load_model(args.model)

    # predict pKa
    ypred = model.predict(Xscale)
    ypred = np.ravel(ypred)

    # save prediction to a file
    df = pd.DataFrame()
    df["PDBID"] = X_topred.index.values
    df["pKa_predict"] = ypred

    df.to_csv(args.out, sep=",", header=True, index=True, float_format="%.3f")

