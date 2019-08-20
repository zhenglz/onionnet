import sys
from sklearn import preprocessing, model_selection
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import argparse
from argparse import RawTextHelpFormatter
import os
from scipy import stats


def rmse(y_true, y_pred):
    dev = np.square(y_true.ravel() - y_pred.ravel())
    return np.sqrt(np.sum(dev) / y_true.shape[0])


def pcc(y_true, y_pred):
    pcc = stats.pearsonr(y_true, y_pred)
    return pcc[0]


def PCC_RMSE(y_true, y_pred):
    alpha = 0.7

    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    rmse = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

    pcc = 1.0 - tf.keras.backend.mean(fsp * fst) / (devP * devT)

    pcc = tf.where(tf.is_nan(pcc), 0.25, pcc)

    return alpha * pcc + (1 - alpha) * rmse


def RMSE(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))


def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    return tf.keras.backend.mean(fsp * fst) / (devP * devT)


def remove_shell_features(dat, shell_index, features_n=64):

    df = dat.copy()

    start = shell_index * features_n
    end = start + features_n

    zeroes = np.zeros((df.shape[0], features_n))

    df[:, start:end] = zeroes

    return df


def remove_atomtype_features(dat, feature_index, shells_n=60):

    df = dat.copy()

    for i in range(shells_n):
        ndx = i * 64 + feature_index

        zeroes = np.zeros(df.shape[0])
        df[:, ndx] = zeroes

    return df


def remove_all_hydrogens(dat, n_features):
    df = dat.copy()

    for f in df.columns.values[:n_features]:
        if "H_" in f or "_H_" in f:
            v = np.zeros(df.shape[0])
            df[f] = v

    return df


def create_model(input_size, lr=0.0001):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, 4, 1, input_shape=input_size))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(64, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Conv2D(128, 4, 1))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(400, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(200,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))

    sgd = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=1e-6, )
    model.compile(optimizer=sgd, loss=PCC_RMSE, metrics=["mse", PCC, RMSE])

    return model


if __name__ == "__main__":
    d = """Train or predict the pKa values based on protein-ligand complexes features.

    Examples:
    python train.py -fn1 docked_training_features_12ksamples_rmsd_lessthan3a.csv 
           -fn2 training_pka_features.csv -fn_val validation_features.csv -history training_hist.csv -pKa_col pKa -train 1

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawTextHelpFormatter)
    parser.add_argument("-fn1", type=str, default=["features_1.csv", ], nargs="+",
                        help="Input. The docked cplx feature set.")
    parser.add_argument("-fn2", type=str, default=["features_2.csv", ], nargs="+",
                        help="Input. The PDBBind feature set.")
    parser.add_argument("-fn_val", type=str, default=["features_3.csv", ], nargs="+",
                        help="Input. The validation feature set.")
    parser.add_argument("-history", type=str, default="history.csv",
                        help="Output. The history information. ")
    parser.add_argument("-pKa_col", type=str, nargs="+", default=["pKa_relu", "pKa_true"],
                        help="Input. The pKa colname as the target. ")
    parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Output. The standard scaler file to save. ")
    parser.add_argument("-model", type=str, default="DNN_Model.h5",
                        help="Output. The trained DNN model file to save. ")
    parser.add_argument("-log", type=str, default="logger.csv",
                        help="Output. The logger file name to save. ")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. The predicted pKa values file name to save. ")
    parser.add_argument("-lr_init", type=float, default=0.001,
                        help="Input. Default is 0.001. The initial learning rate. ")
    parser.add_argument("-epochs", type=int, default=100,
                        help="Input. Default is 100. The number of epochs to train. ")
    parser.add_argument("-batch", type=int, default=128,
                        help="Input. Default is 128. The number of batch size to train. ")
    parser.add_argument("-train", type=int, default=1,
                        help="Input. Default is 1. Whether train or predict. \n"
                             "1: train, 0: predict. ")
    parser.add_argument("-n_features", default=3840, type=int,
                        help="Input. Default is 3840. Number of features in the input dataset.")
    parser.add_argument("-reshape", type=int, default=[64, 60, 1], nargs="+",
                        help="Input. Default is 64 60 1. Reshape the dataset. ")
    parser.add_argument("-remove_H", type=int, default=0,
                        help="Input, optional. Default is 0. Whether remove hydrogens. ")

    args = parser.parse_args()

    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(0)

    X, y = np.array([]), []
    do_eval = False
    ytrue = []

    for fn in args.fn1:
        if os.path.exists(fn):
            #print(fn)
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            print("DataFrame Shape", df.shape, fn)
            if args.train > 0:
                if args.pKa_col[0] in df.columns.values:
                    y = y + list(df[args.pKa_col[0]].values)
                else:
                    print("No such column %s in input file. " % args.pKa_col[0])
            if X.shape[0] == 0:
                X = df.values[:, :args.n_features]
            else:
                X = np.concatenate((X, df.values[:, :args.n_features]), axis=0)

            if args.pKa_col[0] in df.columns.values and args.train == 0:
                ytrue = ytrue + list(df[args.pKa_col[0]].values)

                do_eval = True

    for fn in args.fn2:
        if os.path.exists(fn):

            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            print("DataFrame Shape", df.shape, fn)

            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            X = np.concatenate((X, df.values[:, :args.n_features]), axis=0)
            if args.train > 0:
                y = y + list(df[args.pKa_col[-1]].values)

            if args.pKa_col[-1] in df.columns.values and args.train == 0:
                ytrue = list(ytrue) + df[args.pKa_col[1]].values

                do_eval = True

    col_names = ['pKa', ]
    Xval, yval = np.array([]), []

    for i, fn in enumerate(args.fn_val):

        if os.path.exists(fn):
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            if Xval.shape[0] == 0:
                Xval = df.values[:, :args.n_features]
            else:
                Xval = np.concatenate((Xval, df.values[:, :args.n_features]), axis=0)

            if args.train > 0:
                yval = yval + list(df[col_names[i]].values)

    print(X.shape, len(y), Xval.shape, len(yval))
    print("DataSet Loaded")

    if args.train > 0:

        scaler = preprocessing.StandardScaler()
        Xs = scaler.fit_transform(X)
        Xval = scaler.fit_transform(Xval)
        joblib.dump(scaler, args.scaler)
        print("DataSet Scaled")

        #Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(Xs, y, test_size=0.2)
        #print("Train and test split")
        Xtrain = Xs.reshape((-1, args.reshape[0], args.reshape[1], args.reshape[2]))
        model = create_model((args.reshape[0], args.reshape[1], args.reshape[2]), lr=args.lr_init)

        # callbacks
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=1,
                                                mode='auto', )
        logger = tf.keras.callbacks.CSVLogger(args.log, separator=',', append=False)
        bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath="bestmodel_" + args.model, verbose=1,
                                                       save_best_only=True)

        # train the model
        history = model.fit(Xtrain, y, validation_data=(Xval.reshape(-1, args.reshape[0], args.reshape[1], args.reshape[2]), yval),
                            batch_size=args.batch, epochs=args.epochs, verbose=1, callbacks=[stop, logger, bestmodel])

        model.save(args.model)
        print("Save model. ")

        # np_hist = np.array(history)
        # np.savetxt(args.history, np_hist, delimiter=",", fmt="%.4f")
        # print("Save history.")

    else:
        scaler = joblib.load(args.scaler)

        Xs = scaler.transform(X).reshape((-1, args.reshape[0], args.reshape[1], args.reshape[2]))

        model = tf.keras.models.load_model(args.model,
                                           custom_objects={'RMSE': RMSE,
                                                           'PCC': PCC,
                                                           'PCC_RMSE': PCC_RMSE})

        ypred = pd.DataFrame()
        ypred['pKa_predicted'] = model.predict(Xs).ravel()
        if do_eval:
            print("PCC : %.3f" % pcc(ypred['pKa_predicted'].values, ytrue))
            print("RMSE: %.3f" % rmse(ypred['pKa_predicted'].values, ytrue))

            ypred['pKa_true'] = ytrue

        ypred.to_csv(args.out, header=True, index=True, float_format="%.3f")

