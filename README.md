# Addressing data limitations in seizure prediction through transfer learning
Scripts and convolutional autoencoder model used for applying transfer learning in seizure prediction.

## Authors
- Fábio Lopes
- Mauro F. Pinto
- António Dourado
- Andreas Schulze-Bonhage
- Matthias Dümpelmann
- César Teixeira

## Files

* `test_seizure_prediction_model_tl.py` contains the code necessary to develop patient-specific models (it uses the `seizure_prediction_model_cnn_lstm_autoencoder_128_last_approach.h5` as a transfer learning model). It also uses the statistics (average and standard deviation) from the dataset used to train the transfer learning model (`standardisation_values_cnn_lstm_autoencoder_128_last_approach.npy`).
* `seizure_prediction_model_cnn_lstm_autoencoder_128_last_approach.h5` contains the weights of the transfer learning model in HDF5 file format.
* `standardisation_values_cnn_lstm_autoencoder_128_last_approach.npy` contains the average and standard deviation of the dataset used to train the transfer learning model.
* `utils.py` contains general functions used in the `test_seizure_prediction_model_tl.py`.

## Requirements
* Python 3.7
* Tensorflow 2.6.0
* Numpy 1.19.5
