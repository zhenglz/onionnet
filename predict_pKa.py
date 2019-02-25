import pandas as pd
import numpy as np
from sklearn import preprocessing
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
import tensorflow as tf
from scipy import stats
import os, sys


def PCC_RMSE(y_true, y_pred):
    """
    Custom loss function

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

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
    """
    RMSE metrics

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    """
    dev = np.square(y_true.ravel() - y_pred.ravel())

    return np.sqrt(np.sum(dev) / y_true.shape[0])


def pcc(y_true, y_pred):
    pcc = stats.pearsonr(y_true, y_pred)
    return pcc[0]


def SD(y, y_pred):
    dev = np.square(y.ravel() - y_pred.ravel())

    return np.sqrt(np.sum(dev) / (dev.shape[0] - 1))


def MAE(y, y_pred):
    dev = y.ravel() - y_pred.ravel()
    ave = np.sum(np.absolute(dev)) / dev.shape[0]

    return ave


def RMSE(y_true, y_pred):
#    y_pred = tf.keras.backend.variable(y_pred)
#    y_true = tf.keras.backend.variable(y_true)
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true), axis=-1))

#    with tf.Session() as sess:
#        sess.run(train, {x:x_data, y: y_label}}

def PCC(y_true, y_pred):
    fsp = y_pred - tf.keras.backend.mean(y_pred)
    fst = y_true - tf.keras.backend.mean(y_true)

    devP = tf.keras.backend.std(y_pred)
    devT = tf.keras.backend.std(y_true)

    return tf.keras.backend.mean(fsp * fst) / (devP * devT)


def create_model(dropout=0.5, input_shape=(64, 40, 1), activation="relu"):
    # K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))
    # start a model
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
    model.compile(loss=RMSE, optimizer=sgd)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = tf.keras.callbacks.ModelCheckpoint("parameters.dat", monitor='val_loss', save_weights_only=True,
                                                    verbose=1, save_best_only=True, period=1)

    return model


def remove_shell_features(dat, shell_index, features_n=64):
    df = dat.copy()

    start = shell_index * features_n
    end = start + features_n

    zeroes = np.zeros((df.shape[0], features_n))

    df[:, start:end] = zeroes

    return df


#config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 1})
#sess = tf.Session(config=config)
#tf.set_random_seed(1)
#tf.keras.backend.set_session(sess)

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

# # Scale dataset
scaler = preprocessing.StandardScaler()
X_all = np.concatenate((Xtrain, Xtest, Xval), axis=0)
scaler.fit(X_all)

to_predict = pd.read_csv(sys.argv[2], sep=",", header=0, index_col=0).dropna()
if "pKa" in to_predict.columns.values:
    y_true = to_predict['pKa'].values
    Xpred_scaled = scaler.transform(to_predict.drop(['pKa'], axis=1))
else:
    Xpred_scaled = scaler.transform(to_predict)

val_loss = []

n = 0

Xpred = remove_shell_features(Xpred_scaled, n, 64).reshape((-1, 64, 60, 1))

model = tf.keras.models.load_model(sys.argv[1], custom_objects={'RMSE': RMSE, 'pcc': PCC, 'PCC': PCC, 'PCC_RMSE':PCC_RMSE})
print("Model loaded")

if os.path.exists("parameters.dat"):
    model.load_weights("parameters.dat")
    print("checkpoint_loaded")

y_pred = pd.DataFrame()
y_pred["ID"] = to_predict.index
y_pred["pKa_pred"] = model.predict(Xpred)

if  "pKa" in to_predict.columns.values:
    y_pred["pKa_real"] = y_true
    ypred = y_pred["pKa_pred"]
    print('metrics now ')
    print(rmse(ypred.values, y_true))
    print(pcc(ypred.values, y_true))
    print(MAE(ypred.values, y_true))
    print(SD(ypred.values, y_true))
#    print(tf.to_float(tf.keras.metrics.mean_squared_error(ypred.values, y_real.values)))
    

y_pred.to_csv("predicted_"+sys.argv[2], sep=",", float_format="%.3f", header=True, index=False)


