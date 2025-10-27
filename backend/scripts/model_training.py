# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

def build_lstm(input_shape: tuple) -> tf.keras.Model:
    from tensorflow.keras import layers, models
    inp = tf.keras.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, default="up_1d", choices=["up_1d","up_5d"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--patience", type=int, default=7)
    args = ap.parse_args()

    Xtr = np.load(DATA / "X_train_lstm.npy")
    Xte = np.load(DATA / "X_test_lstm.npy")
    ytr = np.load(DATA / "y_train_lstm.npy")
    yte = np.load(DATA / "y_test_lstm.npy")

    model = build_lstm(Xtr.shape[1:])
    ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=str(MODELS / f"lstm_{args.label}.keras"),
                                              save_best_only=True, monitor="val_loss", mode="min")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=args.epochs, batch_size=args.batch, callbacks=[ckpt, es], verbose=1)

    model.save(MODELS / f"lstm_{args.label}.keras")
    print("Modèle LSTM entraîné et sauvegardé:", MODELS / f"lstm_{args.label}.keras")

if __name__ == "__main__":
    main()
