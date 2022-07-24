import tensorflow as tf
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
import sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import Classification_model
import Preprocessing
import Helper_functions


def run_CNN(X_train, Y_train, X_test, Y_test, VOCABSIZE, EMBEDDING_DIM, MAX_SEN_LENGTH, TEN_FOLD_SIZE, es, cnn_save_to,
            cnn_y_predict):
    cnn_loss = []
    cnn_accuracy = []
    cnn_auc = []
    cnn_precision = []
    cnn_recall = []

    # 10 fold cross validation
    for i in range(10):
        print(i, " iteration begin: ")
        tf.keras.backend.clear_session()
        model_cnn = Classification_model.CNN(VOCABSIZE, EMBEDDING_DIM, MAX_SEN_LENGTH)
        X_train, Y_train = shuffle(X_train, Y_train)
        X_train = X_train[:TEN_FOLD_SIZE]
        Y_train = Y_train[:TEN_FOLD_SIZE]
        history_cnn = model_cnn.fit(X_train, Y_train, batch_size=64, epochs=100, verbose=0, validation_split=0.1,
                                    callbacks=[es])
        score_cnn = model_cnn.evaluate(X_test, Y_test, verbose=0)
        predict_y_cnn = model_cnn.predict(X_test)

        cnn_loss.append(score_cnn[0])
        cnn_accuracy.append(score_cnn[1])
        cnn_auc.append(score_cnn[2])
        cnn_precision.append(score_cnn[3])
        cnn_recall.append(score_cnn[4])

    save_results = {'Precision': cnn_precision, 'Recall': cnn_recall, 'Accuracy': cnn_accuracy, 'AUC': cnn_auc}
    save_results_df = pd.DataFrame(data=save_results)
    save_results_df.to_excel(cnn_save_to)

    predict_y_cnn = predict_y_cnn.flatten()
    save_y_predict = {'Predicted Y_test: ': predict_y_cnn, 'Actual Y_test: ': Y_test}
    save_y_df = pd.DataFrame(data=save_y_predict)
    save_y_df.to_excel(cnn_y_predict)


def run_LSTM(X_train, Y_train, X_test, Y_test, VOCABSIZE, EMBEDDING_DIM, MAX_SEN_LENGTH, TEN_FOLD_SIZE, es,
             lstm_save_to, lstm_y_predict):
    lstm_loss = []
    lstm_accuracy = []
    lstm_auc = []

    lstm_precision = []
    lstm_recall = []

    # 10 fold cross validation
    for i in range(10):
        print(i, " iteration begin: ")
        tf.keras.backend.clear_session()
        model_lstm = Classification_model.LSTM(VOCABSIZE, EMBEDDING_DIM, MAX_SEN_LENGTH)
        X_train, Y_train = shuffle(X_train, Y_train)
        X_train = X_train[:TEN_FOLD_SIZE]
        Y_train = Y_train[:TEN_FOLD_SIZE]

        history_lstm = model_lstm.fit(X_train, Y_train, batch_size=64, epochs=100, verbose=0, validation_split=0.1)
        score_lstm = model_lstm.evaluate(X_test, Y_test, verbose=0)
        predict_y_lstm = model_lstm.predict(X_test)

        lstm_loss.append(score_lstm[0])
        lstm_accuracy.append(score_lstm[1])
        lstm_auc.append(score_lstm[2])
        lstm_precision.append(score_lstm[3])
        lstm_recall.append(score_lstm[4])

    save_results_lstm = {'Precision': lstm_precision, 'Recall': lstm_recall, 'Accuracy': lstm_accuracy, 'AUC': lstm_auc}
    save_results_lstm_df = pd.DataFrame(data=save_results_lstm)
    save_results_lstm_df.to_excel(lstm_save_to)

    predict_y_lstm = predict_y_lstm.flatten()
    save_y_predict = {'Predicted Y_test: ': predict_y_lstm, 'Actual Y_test: ': Y_test}
    save_y_df = pd.DataFrame(data=save_y_predict)
    save_y_df.to_excel(lstm_y_predict)


def run_LGBM(X_train_orig, Y_train, X_test_orig, Y_test, file, TEN_FOLD_SIZE, lgbm_save_to, lgbm_y_predict, es=50):
    vectorizer = TfidfVectorizer(min_df=1, lowercase=True)
    vectorizer.fit(X_train_orig)

    """ use findSet() to limit the number of features to train 
    vocab_set = Helper_functions.findSet(file)
    vectorizer.fit(vocab_set)
    """

    X_train = vectorizer.transform(X_train_orig)
    X_test = vectorizer.transform(X_test_orig)
    y_train = Y_train
    y_test = Y_test

    accuracy = []
    # f1 = []
    auc = []
    precision = []
    recall = []

    for i in range(10):
        print(i, " iteration begin: ")
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)
        X_train = X_train[:TEN_FOLD_SIZE]
        Y_train = Y_train[:TEN_FOLD_SIZE]

        gbm = Classification_model.LGBM(X_train, y_train, es)
        preds = gbm.predict(X_test)
        pred_labels = np.rint(preds)
        pred_labels = [int(x) for x in pred_labels]

        for n in range(len(pred_labels)):
            if pred_labels[n] > 1:
                pred_labels[n] = 1
            if pred_labels[n] < 0:
                pred_labels[n] = 0

        acc_iteration = sklearn.metrics.accuracy_score(y_test, pred_labels)
        # f1_iteration = sklearn.metrics.f1_score(y_test, pred_labels, average='weighted')
        auc_iteration = sklearn.metrics.roc_auc_score(y_test, pred_labels)
        precision_iteration = sklearn.metrics.precision_score(y_test, pred_labels)
        recall_iteration = sklearn.metrics.recall_score(y_test, pred_labels)

        accuracy.append(acc_iteration)
        auc.append(auc_iteration)
        precision.append(precision_iteration)
        recall.append(recall_iteration)

    col1 = "Precision"
    col2 = "Recall"
    col3 = "Accuracy"
    col4 = "AUC"

    data = pd.DataFrame({col1: precision, col2: recall, col3: accuracy, col4: auc})
    writer = pd.ExcelWriter(lgbm_save_to, engine='openpyxl', mode='a')
    data.to_excel(writer, index=False)
    writer.save()

    predict_y_lgbm = pred_labels
    save_y_predict = {'Predicted Y_test: ': predict_y_lgbm, 'Actual Y_test: ': Y_test}
    save_y_df = pd.DataFrame(data=save_y_predict)
    save_y_df.to_excel(lgbm_y_predict)


def run_classification():
    # load in the testing dataset
    df = pd.read_csv('DIRECTORY_PATH_TESTING_DATA')
    # load in the training dataset
    df_t = pd.read_csv('DIRECTORY_PATH_TRAINING_DATA')

    # write the path for where you would like to save accuracy, AUROC, precision, and recall data
    cnn_save_to = "DIRECTORY_PATH" + "cnn_results.xlsx"
    lstm_save_to = "DIRECTORY_PATH" + "lstm_results.xlsx"
    lgbm_save_to = "DIRECTORY_PATH" + "lgbm_results.xlsx"

    # write the path for where you would like to save the predicted tresults for testing data
    cnn_y_predict = "DIRECTORY_PATH" + "cnn_y_predict.xlsx"
    lstm_y_predict = "DIRECTORY_PATH" + "lstm_y_predict.xlsx"
    lgbm_y_predict = "DIRECTORY_PATH" + "lgbm_y_predict.xlsx"

    # process the datasets: discard null data
    df = df[~df.text.isna()]
    df_t = df_t[~df_t.text.isna()]

    # write the phenotype you want to classify
    TARGET_PHENOTYPE = 'PHENOTYPE'
    DATASETS = dict()
    for ds in [TARGET_PHENOTYPE]:
        DATASETS[ds] = Preprocessing.prepare_dataset(df, df_t, ds)

    MAX_SEN_LENGTH = 100
    EMBEDDING_DIM = 50

    X_train, X_test, VOCABSIZE = Preprocessing.prepare_input_for_dl(DATASETS[TARGET_PHENOTYPE]['train'],
                                                                    DATASETS[TARGET_PHENOTYPE]['test'])

    Y_train = DATASETS[TARGET_PHENOTYPE]['y_train']
    Y_test = DATASETS[TARGET_PHENOTYPE]['y_test']

    # find size of the training set
    NUM_TRAIN_SIZE = len(Y_train)
    TEN_FOLD_SIZE = round(NUM_TRAIN_SIZE * 0.9)
    # print("NUM_TRAIN_SIZE: ", NUM_TRAIN_SIZE)
    # print("TEN_FOLD_SIZE: ", TEN_FOLD_SIZE)

    # early stopping
    es = EarlyStopping(monitor='val_loss', min_delta=0, mode="min", verbose=1, patience=5)

    # the vocab file to limit training features for LightGBM
    file = 'VOCAB_FILE_NAME'

    # select which classification model to train
    run_LGBM(X_train, Y_train, X_test, Y_test, file, TEN_FOLD_SIZE, lgbm_save_to, lgbm_y_predict, es)
    run_CNN(X_train, Y_train, X_test, Y_test, VOCABSIZE, EMBEDDING_DIM, MAX_SEN_LENGTH, TEN_FOLD_SIZE, es, cnn_save_to,
            cnn_y_predict)
    run_LSTM(X_train, Y_train, X_test, Y_test, VOCABSIZE, EMBEDDING_DIM, MAX_SEN_LENGTH, TEN_FOLD_SIZE, es,
             lstm_save_to, lstm_y_predict)


if __name__ == '__main__':
    run_classification()
