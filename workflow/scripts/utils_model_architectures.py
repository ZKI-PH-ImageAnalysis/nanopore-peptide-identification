#!/usr/bin/env python

import copy
import os
import random
import numpy as np

def _import_tensorflow_silenced():
    silence = os.environ.get("NP_SILENCE_TF_STARTUP", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if not silence:
        import tensorflow as _tf

        return _tf, _tf.keras.layers

    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        import tensorflow as _tf

        _layers = _tf.keras.layers
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)
    return _tf, _layers


def _set_reproducible_seed(seed, enable_tf_determinism=True):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass
    if not enable_tf_determinism:
        return
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


tf, layers = _import_tensorflow_silenced()


class TimesInceptionBlock(layers.Layer):
    def __init__(self, out_channels, num_kernels=6, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = int(out_channels)
        self.num_kernels = max(1, int(num_kernels))

        kernel_sizes = [1, 3, 5, 7, 9, 11][: self.num_kernels]
        self.branches = [
            layers.Conv2D(
                filters=self.out_channels,
                kernel_size=(1, k),
                padding="same",
                activation=None,
                use_bias=True,
            )
            for k in kernel_sizes
        ]
        self.fuse = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=(1, 1),
            padding="same",
            activation=None,
            use_bias=True,
        )

    def call(self, x, training=False):
        outs = [conv(x) for conv in self.branches]
        x_cat = tf.concat(outs, axis=-1)
        return self.fuse(x_cat)


class TimesBlock(layers.Layer):
    def __init__(
        self,
        seq_len,
        d_model,
        d_ff,
        top_k=3,
        num_kernels=6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.top_k = max(1, int(top_k))
        self.num_kernels = max(1, int(num_kernels))

        self.incep_1 = TimesInceptionBlock(self.d_ff, num_kernels=self.num_kernels)
        self.incep_2 = TimesInceptionBlock(self.d_model, num_kernels=self.num_kernels)

    def _fft_periods(self, x):
        # x: [B, T, C]
        # tf.signal.rfft works on the innermost dimension, so transpose time to the end.
        xf = tf.signal.rfft(tf.transpose(x, [0, 2, 1]))  # [B, C, F]
        amp_global = tf.reduce_mean(tf.abs(xf), axis=[0, 1])
        if self.seq_len > 0:
            amp_global = tf.tensor_scatter_nd_update(amp_global, [[0]], [0.0])

        n_freq = tf.shape(amp_global)[0]
        k_eff = tf.minimum(tf.cast(self.top_k, tf.int32), tf.maximum(n_freq - 1, 1))
        _, top_idx = tf.math.top_k(amp_global, k=k_eff, sorted=True)

        top_idx = tf.cast(top_idx, tf.int32)
        periods = tf.math.floordiv(tf.shape(x)[1], tf.maximum(top_idx, 1))
        periods = tf.maximum(periods, 1)

        amp_batch = tf.reduce_mean(tf.abs(xf), axis=1)  # [B, F]
        period_weight = tf.gather(amp_batch, top_idx, axis=1)
        return periods, period_weight

    def call(self, x, training=False):
        # x: [B, T, C]
        bsz = tf.shape(x)[0]
        tlen = tf.shape(x)[1]
        nchan = tf.shape(x)[2]

        period_list, period_weight = self._fft_periods(x)
        k_eff = tf.shape(period_list)[0]

        outputs = []
        for i in range(self.top_k):
            if i >= int(period_list.shape[0] or self.top_k):
                break

            period = period_list[i]
            pad_mod = tf.math.mod(tlen, period)
            extra = tf.where(tf.equal(pad_mod, 0), 0, period - pad_mod)
            total_len = tlen + extra

            x_pad = tf.pad(x, paddings=[[0, 0], [0, extra], [0, 0]])
            x2 = tf.reshape(x_pad, [bsz, total_len // period, period, nchan])

            y = self.incep_1(x2, training=training)
            y = tf.nn.gelu(y)
            y = self.incep_2(y, training=training)

            y = tf.reshape(y, [bsz, total_len, nchan])
            outputs.append(y[:, :tlen, :])

        if not outputs:
            return x

        out_stack = tf.stack(outputs, axis=-1)  # [B, T, C, K]
        pw = tf.nn.softmax(period_weight[:, : tf.shape(out_stack)[-1]], axis=1)
        pw = tf.reshape(pw, [bsz, 1, 1, tf.shape(out_stack)[-1]])
        out = tf.reduce_sum(out_stack * pw, axis=-1)
        return out + x


class TimesNetClassifier:
    def __init__(
        self,
        n_epochs=200,
        batch_size=128,
        d_model=64,
        d_ff=128,
        e_layers=2,
        top_k=3,
        num_kernels=6,
        dropout=0.1,
        learning_rate=1.0e-3,
        verbose=1,
        callbacks=None,
        loss=None,
        random_state=385,
    ):
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.e_layers = int(e_layers)
        self.top_k = int(top_k)
        self.num_kernels = int(num_kernels)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.verbose = int(verbose)
        self.callbacks = list(callbacks or [])
        self.loss = loss
        self.random_state = int(random_state)

        self.model_ = None
        self.classes_ = None
        self._class_to_index = {}

    @staticmethod
    def _prepare_input(X):
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        elif arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] > 1:
            arr = np.transpose(arr, (0, 2, 1))
        elif arr.ndim != 3:
            raise ValueError(
                f"TimesNetClassifier expects 2D or 3D input, got shape {arr.shape}"
            )
        return arr

    def _encode_y(self, y):
        y_arr = np.asarray(y)
        if self.classes_ is None:
            self.classes_ = np.unique(y_arr)
            self._class_to_index = {c: i for i, c in enumerate(self.classes_)}
        return np.asarray([self._class_to_index[v] for v in y_arr], dtype=np.int32)

    def _build_model(self, seq_len, n_channels, n_classes):
        inp = layers.Input(shape=(seq_len, n_channels), dtype=tf.float32)

        x = layers.Dense(self.d_model, use_bias=True)(inp)
        pos_idx = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = layers.Embedding(input_dim=max(seq_len, 2), output_dim=self.d_model)(
            pos_idx
        )
        x = x + pos_emb

        for _ in range(max(self.e_layers, 1)):
            x = TimesBlock(
                seq_len=seq_len,
                d_model=self.d_model,
                d_ff=self.d_ff,
                top_k=self.top_k,
                num_kernels=self.num_kernels,
            )(x)
            x = layers.LayerNormalization(axis=-1)(x)

        x = layers.Activation(tf.nn.gelu)(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Flatten()(x)
        out = layers.Dense(n_classes, activation="softmax", use_bias=True)(x)

        model = tf.keras.Model(inputs=inp, outputs=out, name="TimesNetClassifier")
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, clipnorm=1.0
        )
        model.compile(
            optimizer=optimizer,
            loss=self.loss or "sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y, X_val=None, y_val=None, callbacks=None, **fit_kwargs):
        _set_reproducible_seed(self.random_state)
        X_in = self._prepare_input(X)
        y_int = self._encode_y(y)

        n_samples, seq_len, n_channels = X_in.shape
        n_classes = len(self.classes_)

        if self.model_ is None:
            self.model_ = self._build_model(
                seq_len=seq_len, n_channels=n_channels, n_classes=n_classes
            )

        cb = list(self.callbacks or [])
        if callbacks:
            cb.extend(list(callbacks))

        use_categorical_targets = callable(self.loss)
        y_fit = (
            tf.keras.utils.to_categorical(y_int, num_classes=n_classes)
            if use_categorical_targets
            else y_int
        )

        val_data = None
        if X_val is not None and y_val is not None:
            Xv = self._prepare_input(X_val)
            yv = np.asarray(
                [self._class_to_index[v] for v in np.asarray(y_val)], dtype=np.int32
            )
            yv_fit = (
                tf.keras.utils.to_categorical(yv, num_classes=n_classes)
                if use_categorical_targets
                else yv
            )
            val_data = (Xv, yv_fit)

        history = self.model_.fit(
            X_in,
            y_fit,
            validation_data=val_data,
            epochs=self.n_epochs,
            batch_size=max(1, self.batch_size),
            verbose=self.verbose,
            callbacks=cb,
            **fit_kwargs,
        )
        return history

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("TimesNetClassifier is not fitted")
        X_in = self._prepare_input(X)
        return self.model_.predict(X_in, verbose=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]


def _clone_optimizer_for_compile(optimizer):
    if optimizer is None:
        return "adam"
    if isinstance(optimizer, str):
        return optimizer
    try:
        cfg = tf.keras.optimizers.serialize(optimizer)
        return tf.keras.optimizers.deserialize(cfg)
    except Exception:
        pass
    try:
        return copy.deepcopy(optimizer)
    except Exception:
        return optimizer


def _clone_callbacks_for_fit(callbacks):
    out = []
    for cb in list(callbacks or []):
        try:
            cfg = tf.keras.callbacks.serialize(cb)
            out.append(tf.keras.callbacks.deserialize(cfg))
            continue
        except Exception:
            pass
        try:
            out.append(copy.deepcopy(cb))
        except Exception:
            out.append(cb)
    return out


def _native_inception_block(x, n_filters, kernel_size, n_conv_per_layer):
    n_branches = max(1, int(n_conv_per_layer))
    ks = [max(2, int(kernel_size // (2**i))) for i in range(n_branches)]
    branches = []
    for k in ks:
        branches.append(
            layers.Conv1D(
                filters=int(n_filters),
                kernel_size=int(k),
                padding="same",
                use_bias=False,
                activation=None,
            )(x)
        )

    pool = layers.MaxPool1D(pool_size=3, strides=1, padding="same")(x)
    pool = layers.Conv1D(
        filters=int(n_filters),
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
    )(pool)
    branches.append(pool)

    x_out = layers.Concatenate(axis=-1)(branches)
    x_out = layers.BatchNormalization()(x_out)
    x_out = layers.Activation("relu")(x_out)
    return x_out


class NativeInceptionTimeClassifier:
    def __init__(
        self,
        n_epochs=200,
        batch_size=128,
        n_classifiers=1,
        n_conv_per_layer=3,
        n_filters=64,
        kernel_size=40,
        depth=4,
        verbose=1,
        optimizer=None,
        callbacks=None,
        loss=None,
        random_state=385,
    ):
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.n_classifiers = int(max(1, n_classifiers))
        self.n_conv_per_layer = int(max(1, n_conv_per_layer))
        self.n_filters = int(max(8, n_filters))
        self.kernel_size = int(max(2, kernel_size))
        self.depth = int(max(1, depth))
        self.verbose = int(verbose)
        self.optimizer = optimizer
        self.callbacks = list(callbacks or [])
        self.loss = loss
        self.random_state = int(random_state)

        self.models_ = []
        self.model_ = None
        self.classes_ = None
        self._class_to_index = {}

    @staticmethod
    def _prepare_input(X):
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        elif arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] > 1:
            arr = np.transpose(arr, (0, 2, 1))
        elif arr.ndim != 3:
            raise ValueError(
                f"NativeInceptionTimeClassifier expects 2D or 3D input, got shape {arr.shape}"
            )
        return arr

    def _encode_y(self, y):
        y_arr = np.asarray(y)
        if self.classes_ is None:
            self.classes_ = np.unique(y_arr)
            self._class_to_index = {c: i for i, c in enumerate(self.classes_)}
        return np.asarray([self._class_to_index[v] for v in y_arr], dtype=np.int32)

    def _build_single_model(self, seq_len, n_channels, n_classes):
        inp = layers.Input(shape=(seq_len, n_channels), dtype=tf.float32)
        x = inp
        residual_anchor = x
        out_channels = self.n_filters * (self.n_conv_per_layer + 1)

        for d in range(self.depth):
            x = _native_inception_block(
                x,
                n_filters=self.n_filters,
                kernel_size=self.kernel_size,
                n_conv_per_layer=self.n_conv_per_layer,
            )
            if d % 3 == 2:
                shortcut = layers.Conv1D(
                    filters=out_channels,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                )(residual_anchor)
                shortcut = layers.BatchNormalization()(shortcut)
                x = layers.Add()([x, shortcut])
                x = layers.Activation("relu")(x)
                residual_anchor = x

        x = layers.GlobalAveragePooling1D()(x)
        out = layers.Dense(n_classes, activation="softmax", use_bias=True)(x)

        model = tf.keras.Model(
            inputs=inp, outputs=out, name="NativeInceptionTimeClassifier"
        )
        model.compile(
            optimizer=_clone_optimizer_for_compile(self.optimizer),
            loss=self.loss or "sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y, X_val=None, y_val=None, callbacks=None, **fit_kwargs):
        _set_reproducible_seed(self.random_state)
        X_in = self._prepare_input(X)
        y_int = self._encode_y(y)
        n_samples, seq_len, n_channels = X_in.shape
        _ = n_samples
        n_classes = len(self.classes_)

        if not self.models_:
            self.models_ = [
                self._build_single_model(
                    seq_len=seq_len, n_channels=n_channels, n_classes=n_classes
                )
                for _ in range(self.n_classifiers)
            ]
            self.model_ = self.models_[0]

        use_categorical_targets = callable(self.loss)
        y_fit = (
            tf.keras.utils.to_categorical(y_int, num_classes=n_classes)
            if use_categorical_targets
            else y_int
        )

        val_data = None
        if X_val is not None and y_val is not None:
            Xv = self._prepare_input(X_val)
            yv = np.asarray(
                [self._class_to_index[v] for v in np.asarray(y_val)], dtype=np.int32
            )
            yv_fit = (
                tf.keras.utils.to_categorical(yv, num_classes=n_classes)
                if use_categorical_targets
                else yv
            )
            val_data = (Xv, yv_fit)

        histories = []
        extra_callbacks = list(callbacks or [])
        for i, mdl in enumerate(self.models_):
            cb = _clone_callbacks_for_fit(self.callbacks)
            cb.extend(_clone_callbacks_for_fit(extra_callbacks))
            hist = mdl.fit(
                X_in,
                y_fit,
                validation_data=val_data,
                epochs=self.n_epochs,
                batch_size=max(1, self.batch_size),
                verbose=self.verbose if i == 0 else 0,
                callbacks=cb,
                **fit_kwargs,
            )
            histories.append(hist)
        return histories[0] if histories else None

    def predict_proba(self, X):
        if not self.models_:
            raise RuntimeError("NativeInceptionTimeClassifier is not fitted")
        X_in = self._prepare_input(X)
        all_probs = [m.predict(X_in, verbose=0) for m in self.models_]
        if len(all_probs) == 1:
            return all_probs[0]
        return np.mean(np.asarray(all_probs), axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.classes_[pred_idx]
