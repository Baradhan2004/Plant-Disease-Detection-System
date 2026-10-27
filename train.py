"""python -m src.train --data data/leaves --arch cnn_rnn   (folder-per-class layout, 70/15/15 stratified split)"""
import argparse, json
from pathlib import Path
import numpy as np


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--arch", default="cnn_rnn")
    ap.add_argument("--img", type=int, default=128); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--out", default="artifacts")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True, parents=True)
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import classification_report, confusion_matrix
    from src.model import build_model
    kw = dict(image_size=(a.img, a.img), batch_size=a.batch, seed=0, validation_split=0.3, label_mode="int")
    tr = keras.utils.image_dataset_from_directory(a.data, subset="training", **kw)
    rest = keras.utils.image_dataset_from_directory(a.data, subset="validation", **kw)
    names = tr.class_names; n_rest = rest.cardinality().numpy()
    va, te = rest.take(n_rest // 2), rest.skip(n_rest // 2)
    tr, va, te = (d.cache().prefetch(tf.data.AUTOTUNE) for d in (tr, va, te))
    m = build_model(len(names), a.arch, a.img); m.summary(); print("parameters:", m.count_params())
    m.fit(tr, validation_data=va, epochs=a.epochs, callbacks=[
        keras.callbacks.EarlyStopping("val_accuracy", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.3, patience=2),
        keras.callbacks.ModelCheckpoint(str(out / f"{a.arch}.keras"), "val_accuracy", save_best_only=True)])
    y = np.concatenate([b[1].numpy() for b in te]); p = m.predict(te).argmax(1)
    rep = classification_report(y, p, target_names=names, output_dict=True)
    json.dump(dict(report=rep, confusion=confusion_matrix(y, p).tolist(), params=m.count_params()), open(out / f"{a.arch}_metrics.json", "w"), indent=2)
    print(classification_report(y, p, target_names=names))
