import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras


# our CNN classification model
def CNN(VOCABSIZE, EMBEDDING_DIM, MAX_SEN_LENGTH):
    activation = "relu"

    model = keras.Sequential()

    model.add(keras.layers.Embedding(input_dim=VOCABSIZE,
                                     output_dim=EMBEDDING_DIM,
                                     input_length=MAX_SEN_LENGTH))

    model.add(keras.layers.Conv1D(filters=32, activation=activation, kernel_size=7))

    model.add(keras.layers.GlobalMaxPool1D())

    model.add(keras.layers.Dense(units=10, activation=activation))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0013), metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model


# our LSTM classification model
def LSTM(VOCABSIZE, EMBEDDING_DIM, MAX_SEN_LENGTH):
    model = keras.Sequential()

    model.add(keras.layers.Embedding(input_dim=VOCABSIZE,
                                     output_dim=EMBEDDING_DIM,
                                     input_length=MAX_SEN_LENGTH))

    model.add(keras.layers.LSTM(50, recurrent_dropout=0))

    model.add(keras.layers.Dense(units=5, activation='relu'))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.00023),
                  metrics=['accuracy', keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model


# our LGBM classification model
def LGBM(X_train, y_train, es):
    dtrain = lgb.Dataset(X_train[100:], label=y_train[100:])
    dval = lgb.Dataset(X_train[:100], label=y_train[:100])

    lgb_clf = lgb
    es_lgbm = lgb.early_stopping(es)

    gbm = lgb_clf.train(
        params={
            'lambda_l1': 0.296,
            'learning_rate': 0.177,
            # 'objective': "binary",
            'objective': "regression",
            'metric': "binary_logloss",
            'num_leaves': 64,
            'max_depth': 74
        },
        train_set=dtrain,
        valid_sets=[dval],
        num_boost_round=10000,
        callbacks=[es_lgbm]
    )

    return gbm