import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import argparse
from argparse import RawDescriptionHelpFormatter
import os
from scipy import stats


def rmse(y_true, y_pred):
    dev = np.square(y_true.ravel() - y_pred.ravel())
    return np.sqrt(np.sum(dev) / y_true.shape[0])


def pcc(y_true, y_pred):
    p = stats.pearsonr(y_true, y_pred)
    return p[0]


def pcc_rmse(y_true, y_pred):
    global alpha

    dev = np.square(y_true.ravel() - y_pred.ravel())
    r = np.sqrt(np.sum(dev) / y_true.shape[0])

    p = stats.pearsonr(y_true, y_pred)[0]

    return (1-p)*alpha + r * (1 - alpha)


def PCC_RMSE(y_true, y_pred):
    global alpha

    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    r = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

    p = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)

    #p = tf.where(tf.is_nan(p), 0.25, p)

    return alpha * p + (1 - alpha) * r


def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    return tf.keras.backend.mean(fsp * fst) / (devP * devT)


def remove_all_hydrogens(dat, n_features):
    df = np.zeros((dat.shape[0], n_features))
    j = -1
    for f in dat.columns.values:
        # remove the hydrogen containing features
        if "H_" not in f and "_H_" not in f and int(f.split("_")[-1]) > 64:
            j += 1
            #if df.shape[0] == 0:
            try:
                df[:, j] = dat[f].values
            except IndexError:
                pass
            print(j, f)

    df = pd.DataFrame(df)
    df.index = dat.index

    return df


if __name__ == "__main__":
    d = """Predict the features based on protein-ligand complexes.

    Examples:
    python predict_pKa.py -fn features_ligands.csv -model ./models/OnionNet_HFree.model \
    -scaler models/StandardScaler.model -out results.csv
    
    
    -fn : containing the features, one sample per row with an ID, 2891 feature values.
    -model: the OnionNet CNN model containing the weights for the networks
    -scaler: the scaler for dataset standardization
    -out: the output pKa, one sample per row with two columns (ID and predicted pKa)

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-fn", type=str, default="features_1.csv",
                        help="Input. The docked cplx feature training set for pKa prediction.")
    parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Output. The standard scaler file to save. ")
    parser.add_argument("-model", type=str, default="DNN_Model.h5",
                        help="Output. The trained DNN model file to save. ")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. The predicted pKa values file name to save. ")


    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)

    scaler = joblib.load(args.scaler)

    Xtest = None

    if os.path.exists(args.fn):
        df = pd.read_csv(args.fn, index_col=0, header=0).dropna()

        Xs = scaler.transform(df.values)
        Xs = pd.DataFrame(Xs)
        Xs.index = df.index
        Xs.columns = df.columns
        Xs = remove_all_hydrogens(Xs, 2891) # 2891 = 49 * 59 * 1
    else:
        print("File not exist: ", args.fn)
        Xs = None
        sys.exit(0)

    print("DataSet Loaded ... ... ")

    # the dataset shape 49*59*1
    Xtest = Xs.values.reshape((-1, 49, 59, 1))

    model = tf.keras.models.load_model(args.model,
                                       custom_objects={'RMSE': RMSE,
                                                       'PCC': PCC,
                                                       'PCC_RMSE': PCC_RMSE})
    ypred = pd.DataFrame(index=Xs.index)
    ypred['pKa_predicted'] = model.predict(Xtest).ravel()
    ypred.to_csv(args.out, header=True, index=True, float_format="%.3f", sep=' ')
    print("pKa Predicted ... ... ")

