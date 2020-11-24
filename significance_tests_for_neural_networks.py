import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
tf.keras.backend.set_floatx('float64')

import sys
import os
if sys.platform == 'darwin':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'#Hacky workaround for an error thrown on macOS.

class ANN(Model):
    def __init__(self, Y_dim):
        super(ANN, self).__init__()
        self.hidden = layers.Dense(25, activation=tf.nn.sigmoid)
        self.out = layers.Dense(Y_dim)

    def call(self, X):
        h = self.hidden(X)
        return self.out(h)

ann = ANN(Y_dim=1)

def mse(y_pred, y_true):
    y_pred = tf.cast(y_pred, tf.float64)
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return tf.reduce_mean(loss)

def get_error(x, y):
    pred = ann(x)
    loss = mse(pred, y)
    return loss

optimizer = tf.keras.optimizers.Adam()#Pretty sure default parameters were used.
            
def run_optimization(x, y):
    with tf.GradientTape() as g:
        pred = ann(x)
        loss = mse(pred, y)

    trainable_variables = ann.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

def generate_data(n, loc, scale):
    epsilon = np.random.normal(loc=loc, scale=scale, size=n)
    X = np.random.uniform(low=-1., high=1., size=[n, 8])
    Y = 8. \
       + np.square(X[:, 0]) \
       + X[:, 1] * X[:, 2] \
       + np.cos(X[:, 3]) \
       + np.exp(X[:, 4] * X[:, 5]) \
       + .1 * X[:, 6] \
       + epsilon
    Y = np.expand_dims(Y, axis=-1)
    assert len(Y.shape) == 2
    return X, Y

loc_noise = 0.
scale_noise = .01
n_train = 100000
n_validation = 10000
n_test = 10000

batch_size = 32
max_epochs = 150

max_steps = int(n_train / batch_size) * max_epochs

X_train, Y_train = generate_data(n_train, loc_noise, scale_noise)
X_validation, Y_validation = generate_data(n_validation, loc_noise, scale_noise)
X_test, Y_test = generate_data(n_test, loc_noise, scale_noise)

train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_data = train_data.repeat().batch(batch_size)

prev_validation_error = np.inf
epoch = 0
early_stopping = 0
for step, (batch_X, batch_Y) in enumerate(train_data.take(max_steps), 1):
    run_optimization(batch_X, batch_Y)

    if step % int(n_train / batch_size) == 0:
        epoch += 1
        validation_error = get_error(X_validation, Y_validation).numpy()
        print('Epoch: %i, step: %i, validation error: %f' % (epoch, step, validation_error))
        early_stopping = early_stopping + 1 if prev_validation_error - validation_error <= 1e-5 else 0
        if early_stopping == 5:
            print('Validation error did not fall below 1e-5 for 5 consecutive epochs; applying early stopping.')
            break
        prev_validation_error = validation_error
print('Done training.')
print('Test error: %f.' % get_error(X_test, Y_test).numpy())
#https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb
