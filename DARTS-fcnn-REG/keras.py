import tensorflow as tf
from tensorflow import keras
import numpy as np

from search import load_data
import matplotlib.pyplot as plt

(train_X, train_y), (valid_X, valid_y) = load_data()
in_dim = np.shape(train_X)[1]
out_dim = np.shape(train_y)[1]

model = keras.Sequential(
    [
        # keras.layers.Flatten(input_shape=(28, 28)),
        # keras.layers.Input((1,)),
        keras.layers.Dense(12),
        keras.layers.Activation("relu"),
        keras.layers.Activation("relu"),
        keras.layers.Dense(20),
        keras.layers.Dense(out_dim),
    ]
)


def r2(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred), axis=0)
    SS_tot = tf.keras.backend.sum(
        tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true, axis=0)), axis=0
    )
    output_scores = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
    r2 = tf.keras.backend.mean(output_scores)
    return r2


model.compile(optimizer="rmsprop", loss="mae", metrics=[r2])

hist = model.fit(
    train_X, train_y, validation_data=(valid_X, valid_y), batch_size=64, epochs=50
)

y_obs = model.predict(valid_X)

# print(hist.history)

plt.subplot(2, 1, 1)
plt.plot(hist.history["r2"], "*-", label="r2")
plt.plot(hist.history["val_r2"], "*-", label="val_r2")
plt.ylim(0, 1)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(valid_X, y_obs, "*", label="pred")
plt.plot(valid_X, valid_y, "*", label="true")
plt.legend()

plt.show()
