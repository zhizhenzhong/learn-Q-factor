import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


from keras.models import load_model

from train import split_sequence, data_process, root_mean_squared_error

NAME = 'RV4'
parser = argparse.ArgumentParser(prog=NAME)
parser.add_argument('-data', type=int, default=1, help='source of data')
parser.add_argument('-redirect', type=int, choices=(0, 1), default=0, help='redirect stdout to logfile')
args = parser.parse_args()

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

_DATA_DIR = os.path.join(os.path.expanduser('~/Desktop/Microsoft_plots_dataset_release/'), 'data-{}'.format(args.data))
_TRAINLOGS_DIR = os.path.join(os.path.expanduser('~/Desktop/Microsoft_plots_dataset_release/'), 'train-logs-{}'.format(args.data))
_MODELS_DIR = os.path.join(os.path.expanduser('~/Desktop/Microsoft_plots_dataset_release/'), 'models-{}'.format(args.data))

def make_test_data(Q_factor_test, sample_step, test_time_scale):
    print('creating testing dataset ...')
    sample_input, sample_output = split_sequence(Q_factor_test, sample_step)
    selected_sample_input = sample_input[0:test_time_scale]
    selected_sample_output = sample_output[0:test_time_scale]
    return selected_sample_input, selected_sample_output


if __name__ == '__main__':
    print('\n===== testing LSTM models in year 2016 =====')

    for channel in range(5,4000):
        for segment in range(1,115):
            _DATA_FILE_NAME = 'channel_' + str(channel) + '_segment_' + str(segment) + '.txt'
            DATA_FILE_PATH = os.path.join(_DATA_DIR, _DATA_FILE_NAME)
            PKL_FILE_PATH = DATA_FILE_PATH.replace('.txt', '.pkl')

            x_train, time_point_train, Q_factor_train, x_test, time_point_test, Q_factor_test = data_process(DATA_FILE_PATH, PKL_FILE_PATH, from_pkl=True, save_pkl=False)

            sample_step = 16
            test_time_scale = 672
            sample_input, sample_output = make_test_data(Q_factor_test, sample_step, test_time_scale)

            #load LSTM models
            load_name = 'LSTM_ch_'+ str(channel) + '_seg_' + str(segment)+ '.h5'
            #load_name = 'multi_Req2Route.h5'
            model_name = os.path.join(_MODELS_DIR, load_name)
            print('\nloading trained model' + model_name + '...')
            model = load_model(model_name, custom_objects={'root_mean_squared_error': root_mean_squared_error})
            #model = load_model(model_name)

            #prediction
            predict_y = []
            loss_y = []

            sample_feature = 1
            sample_input = sample_input.reshape((sample_input.shape[0], sample_input.shape[1], sample_feature))

            for i in range(0, sample_input.shape[0]):
                predict_input = sample_input[i].reshape((1, sample_input.shape[1], sample_feature))
                predict = model.predict(predict_input)
                predict = predict[0][0]
                loss = predict - sample_output[i]

                predict_y.append(predict)
                loss_y.append(loss)

            x = np.array(range(0, len(sample_output)))
            sample_output = np.array(sample_output)
            predict_y = np.array(predict_y)

            ##prediction visualization
            plt.figure()
            plt.plot(x, sample_output, 'b', label='ground truth')
            plt.plot(x, predict_y, 'r', label='LSTM prediction')
            plt.legend(loc='lower left', fontsize=12)
            x_major_locator = MultipleLocator(96)
            ax = plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)
            plt.xlim(0, 672)
            #plt.ylim(14.7, 14.86)
            plt.xlabel("sample data at 15-minutes interval", fontsize=12)
            plt.ylabel("Q-factor", fontsize=12)
            plt.tight_layout()
            plt.savefig("prediction_ch_"+ str(channel) + "_seg_" + str(segment)+ ".pdf", dpi=175)

            # error visualization
            plt.figure()
            plt.plot(x, loss_y)
            plt.xlabel("time", fontsize=12)
            plt.ylabel("Prediction error (predict value minus ground truth)", fontsize=12)
            plt.tight_layout()
            plt.savefig("error_ch_"+ str(channel) + "_seg_" + str(segment)+ ".pdf", dpi=175)

            #cdf of the error
            plt.figure()
            length = len(loss_y)
            loss_y.sort()

            x_cdf = []
            y_cdf = []
            for i in range(length):
                x_cdf.append(float(loss_y[i]))
                y_cdf.append((i + 1) / length)

            plt.plot(x_cdf, y_cdf, c='blue')
            plt.xlabel("prediction error (predict value minus ground truth)", fontsize=12)
            plt.ylabel("Cumulative distribution function", fontsize=12)
            plt.tight_layout()
            plt.savefig("CDF_error_ch_"+ str(channel) + "_seg_" + str(segment)+ ".pdf", dpi=175)
