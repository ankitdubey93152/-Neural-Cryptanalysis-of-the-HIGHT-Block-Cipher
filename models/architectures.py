"""Deep learning neural distinguisher architectures (MLP, 1D CNN, 2D CNN).

All models use:
- ReLU activations for hidden layers
- Sigmoid activation on a single output unit
- Binary cross-entropy loss and Adam optimizer
- Configurable dense layers count for hyperparameter sweeping
"""

from typing import Tuple, Union, Sequence, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


# Default architecture hyperparameters (configurable)
DEFAULT_UNITS: int = 640
DEFAULT_FILTERS_1D: Tuple[int, ...] = (32, 64)
DEFAULT_FILTERS_2D: Tuple[int, ...] = (32, 64)
DEFAULT_KERNEL_SIZE_1D: int = 3
DEFAULT_KERNEL_SIZE_2D: int = 3
DEFAULT_LEARNING_RATE: float = 1e-3
DEFAULT_USE_BATCH_NORM: bool = True
DEFAULT_POOLING: str = "flatten"  # 'flatten' or 'gap' (global average pooling)


def build_mlp(
    input_dim: Union[int, Tuple[int, ...]],
    n_dense_layers: int = 3,
    units: int = DEFAULT_UNITS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """Construct Multi-Layer Perceptron (MLP) distinguisher.

    Args:
        input_dim (int or Tuple[int, ...]): Dimension or shape of flattened input.
        n_dense_layers (int): Number of dense hidden layers (default: 3).
        units (int): Number of units per hidden layer (default: 640).
        learning_rate (float): Adam optimizer learning rate (default: 1e-3).

    Returns:
        tf.keras.Model: Compiled Keras MLP model.
    """
    if isinstance(input_dim, int):
        input_shape = (input_dim,)
    elif len(input_dim) == 1:
        input_shape = input_dim
    else:
        # If multidimensional, flatten first
        input_shape = (int(np.prod(input_dim)),)

    inputs = layers.Input(shape=input_shape, name="mlp_input")
    x = inputs

    for i in range(n_dense_layers):
        x = layers.Dense(units, activation="relu", name=f"dense_{i + 1}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"MLP_L{n_dense_layers}_U{units}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_1d(
    input_shape: Tuple[int, int],
    n_dense_layers: int = 2,
    filters: Sequence[int] = DEFAULT_FILTERS_1D,
    kernel_size: int = DEFAULT_KERNEL_SIZE_1D,
    dense_units: int = DEFAULT_UNITS,
    use_batch_norm: bool = DEFAULT_USE_BATCH_NORM,
    pooling: str = DEFAULT_POOLING,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """Construct 1D Convolutional Neural Network distinguisher.

    Convolves along a single sequence axis.

    Args:
        input_shape (Tuple[int, int]): (sequence_length, channels), e.g. (64, 2).
        n_dense_layers (int): Number of Dense hidden layers before output (default: 2).
        filters (Sequence[int]): Number of filters per Conv1D block (default: (32, 64)).
        kernel_size (int): 1D convolution window size (default: 3).
        dense_units (int): Number of units in Dense layers (default: 640).
        use_batch_norm (bool): Whether to include BatchNormalization (default: True).
        pooling (str): 'flatten' or 'gap' (GlobalAveragePooling1D).
        learning_rate (float): Adam optimizer learning rate (default: 1e-3).

    Returns:
        tf.keras.Model: Compiled Keras 1D CNN model.
    """
    inputs = layers.Input(shape=input_shape, name="cnn1d_input")
    x = inputs

    for i, num_filters in enumerate(filters):
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            name=f"conv1d_{i + 1}",
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)
        x = layers.ReLU(name=f"relu_{i + 1}")(x)

    if pooling == "gap":
        x = layers.GlobalAveragePooling1D(name="gap1d")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    for i in range(n_dense_layers):
        x = layers.Dense(dense_units, activation="relu", name=f"dense_{i + 1}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"CNN1D_L{n_dense_layers}_F{'-'.join(map(str, filters))}",
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_2d(
    input_shape: Tuple[int, int, int],
    n_dense_layers: int = 2,
    kernel_size: int = DEFAULT_KERNEL_SIZE_2D,
    filters: Sequence[int] = DEFAULT_FILTERS_2D,
    dense_units: int = DEFAULT_UNITS,
    use_batch_norm: bool = DEFAULT_USE_BATCH_NORM,
    pooling: str = DEFAULT_POOLING,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tf.keras.Model:
    """Construct 2D Convolutional Neural Network distinguisher.

    Convolves over a 2D-reshaped input with (3x3) filters.

    Args:
        input_shape (Tuple[int, int, int]): (height, width, channels), e.g. (2, 64, 1) or (4, 128, 1).
        n_dense_layers (int): Number of Dense hidden layers before output (default: 2).
        kernel_size (int): 2D convolution kernel size (default: 3 -> 3x3 filter).
        filters (Sequence[int]): Number of filters per Conv2D block (default: (32, 64)).
        dense_units (int): Number of units in Dense layers (default: 640).
        use_batch_norm (bool): Whether to include BatchNormalization (default: True).
        pooling (str): 'flatten' or 'gap' (GlobalAveragePooling2D).
        learning_rate (float): Adam optimizer learning rate (default: 1e-3).

    Returns:
        tf.keras.Model: Compiled Keras 2D CNN model.
    """
    inputs = layers.Input(shape=input_shape, name="cnn2d_input")
    x = inputs

    for i, num_filters in enumerate(filters):
        x = layers.Conv2D(
            filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            name=f"conv2d_{i + 1}",
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)
        x = layers.ReLU(name=f"relu_{i + 1}")(x)

    if pooling == "gap":
        x = layers.GlobalAveragePooling2D(name="gap2d")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    for i in range(n_dense_layers):
        x = layers.Dense(dense_units, activation="relu", name=f"dense_{i + 1}")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"CNN2D_L{n_dense_layers}_K{kernel_size}_F{'-'.join(map(str, filters))}",
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_input(
    X_bits: np.ndarray,
    model_type: str,
    experiment: str
) -> np.ndarray:
    """Format bit tensors to match the expected input shape for each model architecture.

    Shapes produced:
    - Experiment A & C:
      - MLP: (N, 128)
      - CNN_1D: (N, 64, 2)  [sequence=64, channels=2: C1 bits as ch0, C2 bits as ch1]
      - CNN_2D: (N, 2, 64, 1) [row 0 = C1 (64 bits), row 1 = C2 (64 bits), 1 channel]

    - Experiment B:
      - MLP: (N, 512) [4 pairs x 128 bits flattened]
      - CNN_1D: (N, 64, 8) [sequence=64, channels=8: 4 pairs x 2 ciphertexts]
      - CNN_2D: (N, 4, 128, 1) [4 rows (one per pair), 128 bits per row, 1 channel]

    Args:
        X_bits (np.ndarray): Binary bit array.
        model_type (str): 'mlp', 'cnn_1d', or 'cnn_2d'.
        experiment (str): 'a', 'b', or 'c'.

    Returns:
        np.ndarray: Reshaped float32 array ready for training/inference.
    """
    model_type = model_type.lower()
    exp = experiment.lower()
    N = X_bits.shape[0]
    X_float = X_bits.astype(np.float32)

    if exp in ("a", "c"):
        # X_bits shape is (N, 128)
        if model_type == "mlp":
            return X_float.reshape((N, 128))
        elif model_type == "cnn_1d":
            # Reshape into (N, 64, 2): channel 0 is C1 (first 64 bits), channel 1 is C2 (next 64 bits)
            c1_bits = X_float[:, :64]
            c2_bits = X_float[:, 64:]
            return np.stack([c1_bits, c2_bits], axis=-1)  # (N, 64, 2)
        elif model_type == "cnn_2d":
            # Reshape into (N, 2, 64, 1)
            c1_bits = X_float[:, :64]
            c2_bits = X_float[:, 64:]
            grid = np.stack([c1_bits, c2_bits], axis=1)  # (N, 2, 64)
            return np.expand_dims(grid, axis=-1)  # (N, 2, 64, 1)

    elif exp == "b":
        # X_bits shape is (N, 4, 128)
        if model_type == "mlp":
            return X_float.reshape((N, 4 * 128))  # (N, 512)
        elif model_type == "cnn_1d":
            # For 4 pairs: each pair has 2x64 bits -> total 8 channels across 64 length
            # Reshape (N, 4, 2, 64) -> transpose to (N, 64, 8)
            X_pairs = X_float.reshape((N, 4, 2, 64))
            X_trans = np.transpose(X_pairs, (0, 3, 1, 2))  # (N, 64, 4, 2)
            return X_trans.reshape((N, 64, 8))
        elif model_type == "cnn_2d":
            # Reshape into (N, 4, 128, 1)
            return np.expand_dims(X_float, axis=-1)

    raise ValueError(f"Unknown combination of model_type='{model_type}' and experiment='{exp}'")


if __name__ == "__main__":
    print("=" * 60)
    print("SMOKE TEST: BUILDING AND COMPILING ALL THREE MODEL TYPES")
    print("=" * 60)

    # 1. MLP Smoke Test
    print("\n[1/3] MLP Model Summary (input_dim=128, n_dense_layers=3, units=640):")
    mlp = build_mlp(input_dim=128, n_dense_layers=3, units=640)
    mlp.summary()

    # Forward pass check
    dummy_mlp = np.zeros((4, 128), dtype=np.float32)
    out_mlp = mlp(dummy_mlp)
    print(f"MLP Output Shape: {out_mlp.shape}, Sample predictions:\n{out_mlp.numpy().flatten()}")

    # 2. 1D CNN Smoke Test
    print("\n[2/3] 1D CNN Model Summary (input_shape=(64, 2), n_dense_layers=2):")
    cnn1d = build_cnn_1d(input_shape=(64, 2), n_dense_layers=2)
    cnn1d.summary()

    dummy_cnn1d = np.zeros((4, 64, 2), dtype=np.float32)
    out_cnn1d = cnn1d(dummy_cnn1d)
    print(f"1D CNN Output Shape: {out_cnn1d.shape}, Sample predictions:\n{out_cnn1d.numpy().flatten()}")

    # 3. 2D CNN Smoke Test
    print("\n[3/3] 2D CNN Model Summary (input_shape=(2, 64, 1), n_dense_layers=2, kernel_size=3):")
    cnn2d = build_cnn_2d(input_shape=(2, 64, 1), n_dense_layers=2, kernel_size=3)
    cnn2d.summary()

    dummy_cnn2d = np.zeros((4, 2, 64, 1), dtype=np.float32)
    out_cnn2d = cnn2d(dummy_cnn2d)
    print(f"2D CNN Output Shape: {out_cnn2d.shape}, Sample predictions:\n{out_cnn2d.numpy().flatten()}")

    print("\n" + "=" * 60)
    print("ALL MODEL ARCHITECTURES COMPILED AND VERIFIED SUCCESSFULLY!")
    print("=" * 60)
