import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K


from keras.utils import plot_model
from tqdm import tqdm
from keras import layers, callbacks
from keras.models import Sequential
from keras_utils import UpdateMonitor


NAME = 'RV4'
parser = argparse.ArgumentParser(prog=NAME)
parser.add_argument('-data', type=int, default=1, help='source of data')
parser.add_argument('-redirect', type=int, choices=(0, 1), default=0, help='redirect stdout to logfile')
args = parser.parse_args()


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

## plot loss history
class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        #self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        #self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        #self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        #self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        #self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        #self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        #plt.plot(iters, self.accuracy[loss_type], 'r', label='train accuracy')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', linestyle='-', label='overall train loss')
        if loss_type == 'epoch':
            # val_acc
            #plt.plot(iters, self.val_acc[loss_type], 'b', label='validate accuracy')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', linestyle='-', label='overall validate loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="right")


def _tqdm(iterable, desc=None):
    return iterable if args.redirect else tqdm(iterable, desc)

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def data_process(data_file_path, pkl_file_path, from_pkl=False, save_pkl=False):
    if from_pkl:
        if os.path.exists(pkl_file_path):
            print('loading data from', pkl_file_path)
            with open(pkl_file_path, 'rb') as fr:
                x_train, time_point_train, Q_factor_train, x_test, time_point_test, Q_factor_test = pickle.load(fr)
                return x_train, time_point_train, Q_factor_train, x_test, time_point_test, Q_factor_test
        else:
            print('pkl file not found, load from txt file instead')
            return data_process(data_file_path, pkl_file_path, from_pkl=False, save_pkl=True)
    else:
        with open(data_file_path, 'r', encoding='gb2312') as fr:
            time_point_train = []
            Q_factor_train = []
            x_train = []

            time_point_test = []
            Q_factor_test = []
            x_test = []

            index_train = 0
            index_test = 0

            for line in _tqdm(fr, desc='loading'):
                line = line.strip()
                item_time_point = line.split()[0]
                item_Q_factor = line.split()[1]
                #data in year 2015 is set to be training
                if item_time_point[0:4] == '2015':
                    time_point_train.append(item_time_point)
                    item_Q_factor = float(item_Q_factor)
                    Q_factor_train.append(item_Q_factor)
                    x_train.append(index_train)
                    index_train = index_train + 1
                #data in year 2016 is set to be testing data
                else:
                    time_point_test.append(item_time_point)
                    item_Q_factor = float(item_Q_factor)
                    Q_factor_test.append(item_Q_factor)
                    x_test.append(index_test)
                    index_test = index_test + 1

        if save_pkl and not os.path.exists(pkl_file_path):
            with open(pkl_file_path, 'wb') as fw:
                pickle.dump((x_train, time_point_train, Q_factor_train, x_test, time_point_test, Q_factor_test), fw)
            print('routes info saved in', pkl_file_path)
        return x_train, time_point_train, Q_factor_train, x_test, time_point_test, Q_factor_test

def make_train_data(Q_factor_train, sample_step):
    print('creating training dataset ...')
    sample_input, sample_output = split_sequence(Q_factor_train, sample_step)
    return sample_input, sample_output


def model_training(input_data, output_data, n_steps, n_features, channel, segment):

    rnn_size = 64  #32 is good for 96 entry prediction history
    dense_size = 32
    hidden_size = 256
    batch_size = 64
    generative_layers = 1
    dropout_rate = 0.2  #
    reduce_patience = 7
    stop_patience = 8  # early stop
    visualize_num = 20
    epochs = 200

    #LSTM prediction model
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(rnn_size, activation='tanh'), input_shape=(n_steps, n_features)))
    #model.add(layers.GlobalMaxPool1D())
    #model.add(layers.LSTM(rnn_size, activation='relu'))
    #model.add(layers.Dense(dense_size))
    #model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))

    model.compile(optimizer='adam', loss = root_mean_squared_error, metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    # train the model each generation and show predictions against the validation dataset
    reducer = callbacks.ReduceLROnPlateau(patience=reduce_patience, verbose=1)
    stopper = callbacks.EarlyStopping(patience=stop_patience, verbose=1)
    updater = UpdateMonitor()
    plothistory = LossHistory()

    print('start training')
    model.fit(input_data, output_data, epochs=epochs, validation_split=1./5, shuffle=True, callbacks=[updater, reducer, stopper, plothistory])
    print('\ntrain finished\n')

    plothistory.loss_plot('epoch')
    loss_file = 'loss_ch_'+ str(channel) + '_seg_' + str(segment)+ '.pdf'
    loss_location = os.path.join(_MODELS_DIR, loss_file)
    plt.savefig(loss_location, dpi=175)
    _MODELNAME = 'LSTM_ch_'+ str(channel) + '_seg_' + str(segment)+ '.h5'
    model_location = os.path.join(_MODELS_DIR, _MODELNAME)
    model.save(model_location)
    print('\nmodel saved!\n')


_DATA_DIR = os.path.join(os.path.expanduser('~/Desktop/Microsoft_plots_dataset_release/'), 'data-{}'.format(args.data))
_TRAINLOGS_DIR = os.path.join(os.path.expanduser('~/Desktop/Microsoft_plots_dataset_release/'), 'train-logs-{}'.format(args.data))
_MODELS_DIR = os.path.join(os.path.expanduser('~/Desktop/Microsoft_plots_dataset_release/'), 'models-{}'.format(args.data))


if __name__ == '__main__':
    print('\n===== training LSTM models in year 2015 =====')

    _TRAIN_VERBOSE = 2 if args.redirect else 1
    logfile = None
    stdout_bak = sys.stdout

    for channel in range(5,4000):
        for segment in range(1,115):
            _DATA_FILE_NAME = 'channel_' + str(channel) + '_segment_' + str(segment) + '.txt'
            DATA_FILE_PATH = os.path.join(_DATA_DIR, _DATA_FILE_NAME)
            PKL_FILE_PATH = DATA_FILE_PATH.replace('.txt', '.pkl')

            x_train, time_point_train, Q_factor_train, x_test, time_point_test, Q_factor_test = data_process(DATA_FILE_PATH, PKL_FILE_PATH, from_pkl=True, save_pkl=False)

            sample_step = 32
            sample_input, sample_output = make_train_data(Q_factor_train, sample_step)

            sample_feature = 1
            sample_input = sample_input.reshape((sample_input.shape[0], sample_input.shape[1], sample_feature))
            print(sample_input.shape)

            model_training(sample_input, sample_output, sample_step, sample_feature, channel, segment)
