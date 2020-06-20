# %% Import packages
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")
from keras.models import load_model
from keras.optimizers import Adam
import h5py

parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
parser.add_argument('--tracings', default="data/ecg_test.hdf5",  # or date_order.hdf5
                    help='HDF5 containing ecg tracings.')
parser.add_argument('--model', default="dnn_predicts/model.hdf5",  # or model_date_order.hdf5
                    help='file containing training model.')
parser.add_argument('--output_file', default="outputs/dnn_output.npy",  # or predictions_date_order.csv
                    help='output csv file.')
parser.add_argument('-bs', type=int, default=32,
                    help='Batch size.')

args, unk = parser.parse_known_args()
if unk:
    warnings.warn("Unknown arguments:" + str(unk) + ".")

def diagnose(tracings="data/ecg_test.hdf5", model_path='dnn_predicts/model.hdf5', output="outputs/dnn_output.npy", batch_size=32):
    with h5py.File(tracings, "r") as f:
        x = np.array(f['tracings'])
    model = load_model(model_path, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(x, batch_size=batch_size, verbose=1)
#     np.save(args.output_file, y_score)
    return y_score, model
