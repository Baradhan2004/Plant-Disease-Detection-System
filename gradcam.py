"""Grad-CAM for the CNN: which pixels drove the predicted class (requires tensorflow)."""
import numpy as np


def gradcam(model, img, last_conv=None, class_idx=None):
    import tensorflow as tf
    from tensorflow import keras
    last_conv = last_conv or [l.name for l in model.layers if isinstance(l, keras.layers.ReLU)][-1]
    gm = keras.Model(model.inputs, [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        fmap, pred = gm(img[None].astype("float32"))
        cls = int(tf.argmax(pred[0])) if class_idx is None else class_idx
        score = pred[:, cls]
    g = tape.gradient(score, fmap); w = tf.reduce_mean(g, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap * w[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-9), cls
