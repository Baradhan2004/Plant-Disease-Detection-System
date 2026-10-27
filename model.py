"""Keras models (tensorflow>=2.15; not installed in the authoring sandbox, see README).

cnn      : 4 conv blocks -> GAP -> dense softmax
cnn_rnn  : the same conv stack, but the final feature map (H x W x C) is read as a sequence of H rows
           (each row = W*C features) and passed through a Bidirectional GRU, so the model can reason about
           how lesion patterns are laid out spatially from the leaf tip to the stem.
"""


def conv_stack(x, widths=(32, 64, 128, 128)):
    from tensorflow.keras import layers
    for w in widths:
        x = layers.Conv2D(w, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(w, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)
    return x


def build_model(n_classes, arch="cnn_rnn", img_size=128, rnn_units=96):
    from tensorflow import keras
    from tensorflow.keras import layers
    inp = keras.Input((img_size, img_size, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inp)
    x = layers.RandomRotation(0.1)(x); x = layers.RandomZoom(0.1)(x); x = layers.RandomContrast(0.15)(x)
    x = layers.Rescaling(1 / 255.0)(x)
    x = conv_stack(x)                                   # -> (img/16, img/16, 128)
    if arch == "cnn_rnn":
        h, w, c = x.shape[1], x.shape[2], x.shape[3]
        x = layers.Reshape((h, w * c))(x)               # H timesteps, W*C features each
        x = layers.Bidirectional(layers.GRU(rnn_units, dropout=0.2))(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out, name=arch)
    m.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy",
              metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(3, name="top3")])
    return m
