"""
Defining the model.
"""
import tensorflow as tf
import wandb


def get_next_dim(current_dim: int, padding: int, kernel_size: int, stride: int) -> int:
    """
    Calculates the new dimension for an output volume after applying a
    convolutional layer with the given dimensions.
    """
    return int((current_dim + 2 * padding - kernel_size) / (stride) + 1)


def make_cnn(
    image_size: int,
    n_classes: int,
    *,
    n_input_channels: int = 3,
    initial_filters: int = 16,
    max_filters: int = 32,
    kernel_size: int = 5,
    stride: int = 2,
    min_volume_size: int = 3,
    l2_regularization: float = 0.0,
    dropout_rate: float = 0.0,
):
    """Builds a CNN-based classification model.

    Parameters
    ----------
    image_size : int
        The size of an input image along one side in pixels. It is assumed
        the image is square.
    n_classes : int
        The number of classes the model is trying to predict.
    n_input_channels : int, optional
        The number of channels each input image has.
    initial_filters : int, optional
        The number of filters (i.e. feature maps or hidden channels) to use
        on the first convolutional layer.
    max_filters : int, optional
        The maximum number of filters to use in any given convolutional layer.
    kernel_size : int, optional
        The size of the kernal to use when convolving.
    stride : int, optional
        The stride to use when convolving.
    min_volume_size : int, optional
        The minimum allowable size of the first two dimensions of a convolutional
        layer's output volume. The volume will not become smaller than
        `(min_volume_size, min_volume_size, filters)`.
    l2_regularization : float, optional
        Value for the lambda parameter of L2 regularization, applied to every
        layer.
    dropout_rate : float, optional
        Dropout rate to use. Applied to every convolutional and fully connected
        layer (except the output layer).
    """
    wandb.config.update(
        {
            "n_input_channels": n_input_channels,
            "initial_filters": initial_filters,
            "max_filters": max_filters,
            "kernel_size": kernel_size,
            "stride": stride,
            "min_volume_size": min_volume_size,
            "l2_regularization": l2_regularization,
            "dropout_rate": dropout_rate,
        }
    )

    volume_size = image_size
    filters = initial_filters

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    X = inputs

    i = 0
    while get_next_dim(volume_size, 0, kernel_size, stride) >= min_volume_size:
        X = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="valid",
            name=f"conv2D_{i}",
        )(X)
        X = tf.keras.layers.BatchNormalization(name=f"batchnorm_{i}")(X)
        X = tf.keras.layers.Activation("relu", name=f"relu_{i}")(X)
        X = tf.keras.layers.Dropout(dropout_rate)(X)

        volume_size = get_next_dim(volume_size, 0, kernel_size, stride)
        filters = min(filters * 2, max_filters)
        i += 1

    X = tf.keras.layers.Flatten()(X)

    # One FC hidden layer.
    X = tf.keras.layers.Dense(128, name=f"dense_{i}")(X)
    X = tf.keras.layers.BatchNormalization(name=f"batchnorm_{i}")(X)
    X = tf.keras.layers.Activation("relu", name=f"relu_{i}")(X)
    X = tf.keras.layers.Dropout(dropout_rate)(X)
    i += 1

    # Final output layer.
    X = tf.keras.layers.Dense(n_classes, name=f"dense_{i}")(X)
    Y_hat = tf.keras.layers.Activation("softmax", name=f"softmax_{i}")(X)

    model = tf.keras.Model(inputs=inputs, outputs=Y_hat)

    # Add optional l2 regularization to each layer of the model
    regularizer = tf.keras.regularizers.l2(l2_regularization)
    for layer in model.layers:
        if hasattr(layer, "kernel_regularizer") and layer.trainable:
            setattr(layer, "kernel_regularizer", regularizer)

    return model
