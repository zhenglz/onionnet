import pandas as pd
import numpy as np
from sklearn import preprocessing
from tensorflow.python.client import device_lib
import tensorflow as tf
from scipy import stats
import itertools

import os, sys
#os.environ["CUDA_DEVICE_ORDER"] = "00000000:02:00.0"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def create_model(dropout=0.5, input_shape=(64, 40, 1)):
    model = tf.keras.models.Sequential()

    # first convolution 128
    model.add(tf.keras.layers.Conv2D(64, kernel_size=4, kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                     input_shape=input_shape))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    # pooling2D
    '''model.add(tf.keras.layers.MaxPooling2D(
        pool_size=4,
        strides=4,
        padding='same',    # Padding method
    ))'''

    # convolution 64
    model.add(tf.keras.layers.Conv2D(32, kernel_size=4, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    # pooling2D
    '''model.add(tf.keras.layers.MaxPooling2D(
        pool_size=3,
        strides=3,
        padding='same',    # Padding method
    ))'''

    # convolution 32
    model.add(tf.keras.layers.Conv2D(16, kernel_size=4, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    # pooling 2d
    '''model.add(tf.keras.layers.MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',    # Padding method
    ))'''

    # 3 Dense layers with dropout
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Activation("relu"))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Activation("relu"))
    # model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dropout(dropout))

    # model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Activation("relu"))

    # compile model using accuracy to measure model performance
    sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1, clipvalue=0.5)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(dropout=0.2, MAX_STEPS=200, bsize=32, input_shape=(64, 60, 1), log="log.log"):
    model = create_model(dropout=dropout, input_shape=input_shape)

    training_results = []

    for n in range(1, MAX_STEPS):
        model.fit(Xtrain_scaled, ytrain.values, batch_size=bsize, epochs=1)
        if n % 10 == 0:
            model.save("model_dout%.1f_epoch%d_bsize%d.h5"%(dropout, n, bsize))

        yp = model.predict(Xtrain_scaled)
        pcc_loss = PCC(np.ravel(yp), ytrain.values)[0]
        rmse_loss = RMSE(np.ravel(yp), ytrain.values)

        print("Current round %d "%n)
        ytest_pred = model.predict(Xtest_scaled)
        rmse = RMSE(np.ravel(ytest_pred), ytest.values)
        pcc = PCC(np.ravel(ytest_pred), ytest.values)[0]

        yval_pred = model.predict(Xvalidate_scaled)
        rmse2 = RMSE(np.ravel(yval_pred), yval.values)
        pcc2 = PCC(np.ravel(yval_pred), yval.values)[0]

        print("Train: %.3f %.3f "%(rmse_loss, pcc_loss))
        print("Test : %.3f %.3f "%(rmse, pcc))
        print("Valid: %.3f %.3f "%(rmse2, pcc2))

        r = [n, rmse_loss, pcc_loss, rmse, pcc, rmse2, pcc2]
        training_results.append(r)

        if n % 20 == 0:
            df = pd.DataFrame(training_results)
            df.columns = ["epoch", "train_rmse", "train_pcc", "test_rmse", "test_pcc", "validate_rmse", "validate_pcc"]
            df.to_csv(log, sep=",", header=True, index=False, float_format="%.3f")


config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 1})
sess = tf.Session(config=config)

tf.set_random_seed(1)
tf.keras.backend.set_session(sess)

train = pd.read_csv("training_pka_features.csv", sep=",", header=0, index_col=0)

train = train.dropna()
Xtrain = train.iloc[:, :-1]
ytrain = train['pKa']

test = pd.read_csv("testing_pka_features.csv", sep=",", header=0, index_col=0)
test = test.dropna()
Xtest = test.iloc[:, :-1]
ytest = test['pKa']

validate = pd.read_csv("validateset_pka_features.csv", sep=",", header=0, index_col=0)
validate = validate.dropna()
Xval = validate.iloc[:, :-1]
yval = validate['pKa']

original_features = Xtrain.columns.values

i=0
remove_ndx = range(i*64, i*64+64)
remove_features = [str(x) for x in remove_ndx]

keep_features = [x for x in original_features if (x.split("_")[2] not in remove_features)]

kf = [x for x in keep_features if (x.split("_")[0] != "H" and x.split("_")[1] != "H")]
keep_features = kf

print(remove_features, remove_ndx)
print(keep_features)


X_train = Xtrain[keep_features]
X_test  = Xtest[keep_features]
X_val = Xval[keep_features]

# # Scale dataset
scaler = preprocessing.StandardScaler()

X_all = np.concatenate((X_train, X_test, X_val), axis=0)
print("X_ALL SIZE" * 10, X_all.shape)
scaler.fit(X_all)

input_shape = (-1, 49, 59, 1)

Xtrain_scaled = scaler.transform(X_train).reshape(input_shape)
Xtest_scaled = scaler.transform(X_test).reshape(input_shape)
Xvalidate_scaled = scaler.transform(X_val).reshape(input_shape)

X_topred = pd.read_csv(sys.argv[2], sep=",", header=0, index_col=0)
print(X_topred.shape)
Xscale = scaler.transform(X_topred[keep_features]).reshape(input_shape)

# model = create_model(dropout=dropout, input_shape=(64-len(exclude_pairs), 60, 1))
model = tf.keras.models.load_model(sys.argv[1])

ypred = model.predict(Xscale)

ypred = np.ravel(ypred)

df = pd.DataFrame()
df["PDBID"] = X_topred.index.values
df["pKa_predict"] = ypred

if "pKa" in X_topred.columns.values:
    df["pKa_real"] = X_topred['pKa']
    print(RMSE(ypred, X_topred['pKa']))
    print(PCC(ypred, X_topred['pKa']))
    print(MAE(ypred, X_topred['pKa']))
    print(SD(ypred, X_topred['pKa']))

df.to_csv("predicted_noHnoshell0" + sys.argv[2], sep=",", header=True, index=True, float_format="%.3f")


