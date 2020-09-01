import sys
from sklearn import preprocessing, model_selection
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import argparse
from argparse import RawTextHelpFormatter, RawDescriptionHelpFormatter
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


def create_model(input_size, lr=0.0001, maxpool=True, dropout=0.1):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=4, strides=1,
                                     padding="valid", input_shape=input_size))
    model.add(tf.keras.layers.Activation("relu"))
    if maxpool:
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
        ))

    model.add(tf.keras.layers.Conv2D(64, 4, 1, padding="valid"))
    model.add(tf.keras.layers.Activation("relu"))
    if maxpool:
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
        ))

    model.add(tf.keras.layers.Conv2D(128, 4, 1, padding="valid"))
    model.add(tf.keras.layers.Activation("relu"))
    if maxpool:
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
        ))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(400, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(200,
                                    kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(100, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(dropout))

    #model.add(tf.keras.layers.Dense(20, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    #model.add(tf.keras.layers.Activation("relu"))
    #model.add(tf.keras.layers.BatchNormalization())
    #model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), ))
    model.add(tf.keras.layers.Activation("relu"))

    sgd = tf.keras.optimizers.SGD(lr=lr, momentum=0.9, decay=1e-6, )
    model.compile(optimizer=sgd, loss=PCC_RMSE, metrics=['mse'])

    return model


if __name__ == "__main__":
    d = """Train or predict the features based on protein-ligand complexes.

    Examples:
    python CNN_model_keras.py -fn1 docked_training_features_12ksamples_rmsd_lessthan3a.csv 
           -fn2 training_pka_features.csv -history hist.csv -pKa_col pKa_mimic pKa -train 1

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-fn_train", type=str, default=["features_1.csv", ], nargs="+",
                        help="Input. The docked cplx feature training set.")
    parser.add_argument("-fn_validate", type=str, default=["features_2.csv", ], nargs="+",
                        help="Input. The PDBBind feature validating set.")
    parser.add_argument("-fn_test", type=str, default=["features_2.csv", ], nargs="+",
                        help="Input. The PDBBind feature testing set.")
    parser.add_argument("-y_col", type=str, nargs="+", default=["pKa_relu", "pKa_true"],
                        help="Input. The pKa colname as the target. ")
    parser.add_argument("-scaler", type=str, default="StandardScaler.model",
                        help="Output. The standard scaler file to save. ")
    parser.add_argument("-prev_scaler", type=str, default="model/prev_scaler.lib", 
                        help="Load previously trained scaler.")
    parser.add_argument("-model", type=str, default="DNN_Model.h5",
                        help="Output. The trained DNN model file to save. ")
    parser.add_argument("-prev_model", type=str, default="model/prev_trained_model.h5", 
                        help="Load previously trained model to fine tune the model.")
    parser.add_argument("-log", type=str, default="",
                        help="Output. The logger file name to save. ")
    parser.add_argument("-out", type=str, default="predicted_pKa.csv",
                        help="Output. The predicted pKa values file name to save. ")
    parser.add_argument("-lr_init", type=float, default=0.001,
                        help="Input. Default is 0.001. The initial learning rate. ")
    parser.add_argument("-epochs", type=int, default=100,
                        help="Input. Default is 100. The number of epochs to train. ")
    parser.add_argument("-batch", type=int, default=128,
                        help="Input. Default is 128. The batch size. ")
    parser.add_argument("-patience", type=int, default=40,
                        help="Input. Default is 40. The patience steps. ")
    parser.add_argument("-delta_loss", type=float, default=0.01,
                        help="Input. Default is 0.01. The delta loss for early stopping. ")
    parser.add_argument("-dropout", type=float, default=0.1,
                        help="Input. Default is 0.1. The dropout rate. ")
    parser.add_argument("-alpha", type=float, default=0.1,
                        help="Input. Default is 0.1. The alpha value. ")
    parser.add_argument("-train", type=int, default=1,
                        help="Input. Default is 1. Whether train or predict. \n"
                             "1: train, 0: predict. ")
    parser.add_argument("-pooling", type=int, default=0,
                        help="Input. Default is 0. Whether using maxpooling. \n"
                             "1: with pooling, 0: no pooling. ")
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

    X, y = None, []
    do_eval = False

    global alpha
    alpha = args.alpha

    for i, fn in enumerate(args.fn_train):
        if os.path.exists(fn):
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            print("DataFrame Shape", df.shape)

            if args.train:
                if args.y_col[0] in df.columns.values:
                    y = y + list(df[args.y_col[0]].values)
                else:
                    print("No such column %s in input file. " % args.y_col[0])

            if i == 0:
                X = df.values[:, :args.n_features]
            else:
                X = np.concatenate((X, df.values[:, :args.n_features]), axis=0)

    Xval, yval = None, []
    for i, fn in enumerate(args.fn_validate):
        if os.path.exists(fn):
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            if i == 0:
                Xval = df.values[:, :args.n_features]
            else:
                Xval = np.concatenate((Xval, df.values[:, :args.n_features]), axis=0)

            if args.train:
                yval = yval + list(df[args.y_col[-1]].values)

    Xtest, ytest = None, []
    for i, fn in enumerate(args.fn_test):
        if os.path.exists(fn):
            df = pd.read_csv(fn, index_col=0, header=0).dropna()
            if args.remove_H:
                df = remove_all_hydrogens(df, args.n_features)

            if i == 0:
                Xtest = df.values[:, :args.n_features]
            else:
                Xtest = np.concatenate((Xtest, df.values[:, :args.n_features]), axis=0)

            if args.train:
                ytest = ytest + list(df[args.y_col[-1]].values)

    print("DataSet Loaded")

    if args.train > 0:

        if not os.path.exists(args.prev_scaler):
            scaler = preprocessing.StandardScaler()
            X_train_val = np.concatenate((X, Xval), axis=0)
            scaler.fit(X_train_val)
        else:
            scaler = joblib.load(args.scaler)
            
        joblib.dump(scaler, args.scaler)

        Xtrain = scaler.transform(X).reshape((-1, args.reshape[0],
                                              args.reshape[1],
                                              args.reshape[2]))
        Xval = scaler.transform(Xval).reshape((-1, args.reshape[0],
                                               args.reshape[1],
                                               args.reshape[2]))
        Xtest = scaler.transform(Xtest).reshape((-1, args.reshape[0],
                                                 args.reshape[1],
                                                 args.reshape[2]))
        ytrain = np.array(y).reshape((-1, 1))
        yval = np.array(yval).reshape((-1, 1))
        ytest = np.array(ytest).reshape((-1, 1))

        print("DataSet Scaled")

        if not os.path.exists(args.prev_model):
            model = create_model((args.reshape[0], args.reshape[1], args.reshape[2]),
                                 lr=args.lr_init, dropout=args.dropout, maxpool=args.pooling)
        else:
            model = tf.keras.models.load_model(args.model,
                                               custom_objects={'RMSE': RMSE,
                                                               'PCC': PCC,
                                                               'PCC_RMSE': PCC_RMSE})

        stopping = [[0, 999.9], ]
        history = []

        # train the model
        for e in range(1, args.epochs+1):
            model.fit(Xtrain, ytrain, validation_data=(Xval, yval),
                      batch_size=args.batch, epochs=1, verbose=1)

            ytrain_pred = model.predict(Xtrain).ravel()
            loss = pcc_rmse(ytrain.ravel(), ytrain_pred)
            pcc_train = pcc(ytrain.ravel(), ytrain_pred)
            rmse_train = rmse(ytrain.ravel(), ytrain_pred)

            yval_pred = model.predict(Xval).ravel()
            loss_val = pcc_rmse(yval.ravel(), yval_pred)
            pcc_val = pcc(yval.ravel(), yval_pred)
            rmse_val = rmse(yval.ravel(), yval_pred)

            ytest_pred = model.predict(Xtest).ravel()
            loss_test = pcc_rmse(ytest.ravel(), ytest_pred)
            pcc_test = pcc(ytest.ravel(), ytest_pred)
            rmse_test = rmse(ytest.ravel(), ytest_pred)

            history.append([e, loss, pcc_train, rmse_train,
                            loss_val, pcc_val, rmse_val,
                            loss_test, pcc_test, rmse_test])
            hist    = pd.DataFrame(history, columns=['epoch', 'loss', 'pcc_train', 'rmse_train',
                                                     'loss_val', 'pcc_val', 'rmse_val',
                                                     'loss_test', 'pcc_test', 'rmse_test'])

            if args.log == "":
                log = "log_batch%d_dropout%.1f_alpha%.1f_withH%d.csv" % \
                (args.batch, args.dropout, args.alpha, args.remove_H)
            else:
                log = args.log

            hist.to_csv(log, header=True, index=False, sep=",", float_format="%.4f")
            print("EPOCH:%d Loss:%.3f RMSE:%.3f PCC:%.3f LOSS_VAL:%.3f RMSE:%.3f PCC:%.3f LOSS_TEST:%.3f RMSE_TEST:%.3f PCC_TEST:%.3f"%
                  (e, loss, rmse_train, pcc_train, loss_val, rmse_val, pcc_val, loss_test, rmse_test, pcc_test ))            

            if stopping[-1][1] - loss_val >= args.delta_loss:
                print("Model improve from %.3f to %.3f. Save model to %s."
                      % (stopping[-1][1], loss_val, args.model))

                model.save(args.model)
                stopping.append([e, loss_val])
            else:
                if e - stopping[-1][0] >= args.patience:
                    print("Get best model at epoch = %d." % stopping[-1][0])
                    break
                else:
                    pass

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
            print("PCC : %.3f" % pcc(ypred['pKa_predicted'].values, ytest))
            print("RMSE: %.3f" % rmse(ypred['pKa_predicted'].values, ytest))

            ypred['pKa_true'] = ytest

        ypred.to_csv(args.out, header=True, index=True, float_format="%.3f")

