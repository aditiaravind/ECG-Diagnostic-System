from keras.layers import (Input, Conv1D, MaxPooling1D, Dropout,
                          BatchNormalization, Activation, Add,
                          Flatten, Dense)
from keras.models import Model
import numpy as np
import io

class ResidualUnit(object):
    
    def __init__(self, n_samples_out, n_filters_out, kernel_initializer='he_normal',
                 dropout_rate=0.8, kernel_size=17, preactivation=True,
                 postactivation_bn=False, activation_function='relu'):
        self.n_samples_out = n_samples_out
        self.n_filters_out = n_filters_out
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.preactivation = preactivation
        self.postactivation_bn = postactivation_bn
        self.activation_function = activation_function

    def _skip_connection(self, y, downsample, n_filters_in):
        """Implement skip connection."""
        # Deal with downsampling
        if downsample > 1:
            y = MaxPooling1D(downsample, strides=downsample, padding='same')(y)
        elif downsample == 1:
            y = y
        else:
            raise ValueError("Number of samples should always decrease.")
        # Deal with n_filters dimension increase
        if n_filters_in != self.n_filters_out:
            # This is one of the two alternatives presented in ResNet paper
            # Other option is to just fill the matrix with zeros.
            y = Conv1D(self.n_filters_out, 1, padding='same',
                       use_bias=False, kernel_initializer=self.kernel_initializer)(y)
        return y

    def _batch_norm_plus_activation(self, x):
        if self.postactivation_bn:
            x = Activation(self.activation_function)(x)
            x = BatchNormalization(center=False, scale=False)(x)
        else:
            x = BatchNormalization()(x)
            x = Activation(self.activation_function)(x)
        return x

    def __call__(self, inputs):
        """Residual unit."""
        x, y = inputs
        n_samples_in = y.shape[1].value
        downsample = n_samples_in // self.n_samples_out
        n_filters_in = y.shape[2].value
        y = self._skip_connection(y, downsample, n_filters_in)
        # 1st layer
        x = Conv1D(self.n_filters_out, self.kernel_size, padding='same',
                   use_bias=False, kernel_initializer=self.kernel_initializer)(x)
        x = self._batch_norm_plus_activation(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)

        # 2nd layer
        x = Conv1D(self.n_filters_out, self.kernel_size, strides=downsample,
                   padding='same', use_bias=False,
                   kernel_initializer=self.kernel_initializer)(x)
        if self.preactivation:
            x = Add()([x, y])  # Sum skip connection and main connection
            y = x
            x = self._batch_norm_plus_activation(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        else:
            x = BatchNormalization()(x)
            x = Add()([x, y])  # Sum skip connection and main connection
            x = Activation(self.activation_function)(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
            y = x
        return [x, y]


# ----- Model ----- #
kernel_size = 16
kernel_initializer = 'he_normal'
signal = Input(shape=(4096, 12), dtype=np.float32, name='signal')
age_range = Input(shape=(6,), dtype=np.float32, name='age_range')
is_male = Input(shape=(1,), dtype=np.float32, name='is_male')
x = signal
x = Conv1D(64, kernel_size, padding='same', use_bias=False,
           kernel_initializer=kernel_initializer)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x, y = ResidualUnit(1024, 128, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, x])
x, y = ResidualUnit(256, 196, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
x, y = ResidualUnit(64, 256, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
x, _ = ResidualUnit(16, 320, kernel_size=kernel_size,
                    kernel_initializer=kernel_initializer)([x, y])
x = Flatten()(x)
diagn = Dense(6, activation='sigmoid', kernel_initializer=kernel_initializer)(x)
model = Model(signal, diagn)
# ----------------- #


if __name__ == "__main__":
    model.summary()