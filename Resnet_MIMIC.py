import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model

def resnet_block(input_tensor, filters, kernel_size=3, strides=1, use_batch_norm=True):
    """
    A single ResNet block with residual connections.
    """
    # First convolutional layer
    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer="he_normal")(input_tensor)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second convolutional layer
    x = Conv1D(filters, kernel_size=kernel_size, strides=1, padding="same", kernel_initializer="he_normal")(x)
    if use_batch_norm:
        x = BatchNormalization()(x)

    # Residual connection
    if strides != 1 or input_tensor.shape[-1] != filters:
        # Adjust the input tensor dimensions to match the output dimensions
        input_tensor = Conv1D(filters, kernel_size=1, strides=strides, padding="same", kernel_initializer="he_normal")(input_tensor)
        if use_batch_norm:
            input_tensor = BatchNormalization()(input_tensor)
    
    x = Add()([x, input_tensor])
    x = Activation("relu")(x)

    return x

def ResNet1D(input_shape, num_blocks, filters, num_outputs=2):
    """
    ResNet architecture for 1D time-series data.

    Args:
        input_shape: Shape of the input data (e.g., (875, 1) for the PPG dataset).
        num_blocks: Number of ResNet blocks to stack in each stage.
        filters: List of filters for each stage.
        num_outputs: Number of output nodes (e.g., 2 for SBP and DBP).

    Returns:
        model: A Keras Model instance.
    """
    X_input = Input(shape=input_shape)

    # Initial Conv layer
    x = Conv1D(filters[0], kernel_size=7, strides=2, padding="same", kernel_initializer="he_normal")(X_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # ResNet blocks
    for i, f in enumerate(filters):
        for j in range(num_blocks[i]):
            strides = 2 if j == 0 and i > 0 else 1  # Downsample only at the start of each stage
            x = resnet_block(x, f, strides=strides)

    # Global average pooling
    x = GlobalAveragePooling1D()(x)

    # Fully connected layers for SBP and DBP
    SBP = Dense(1, activation="relu", name="SBP")(x)
    DBP = Dense(1, activation="relu", name="DBP")(x)

    model = Model(inputs=X_input, outputs=[SBP, DBP], name="ResNet1D")

    return model

# Define the model parameters
input_shape = (875, 1)  # PPG time series: 875 data points, 1 channel
num_blocks = [2, 2, 2]  # Number of blocks per stage
filters = [64, 128, 256]  # Filters for each stage
num_outputs = 2  # SBP and DBP

# Create the ResNet model
model = ResNet1D(input_shape, num_blocks, filters, num_outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={"SBP": "mse", "DBP": "mse"},
    metrics={"SBP": "mae", "DBP": "mae"}
)

# Display the model summary
model.summary()

history = model.fit(
    x=ppg_data, 
    y={"SBP": sbp_labels, "DBP": dbp_labels},
    validation_split=0.2,
    epochs=50,
    batch_size=32
)
