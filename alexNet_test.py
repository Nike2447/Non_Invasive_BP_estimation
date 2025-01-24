from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, BatchNormalization, Flatten, Dropout, Activation
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model
import tensorflow as tf
import h5py
import numpy as np

# Load dataset
def load_dataset(filename, num_samples=None):
    with h5py.File(filename, 'r') as h5:
        ppg = h5['ppg'][:num_samples]
        labels = h5['label'][:num_samples]
        subject_idx = h5['subject_idx'][:num_samples]
    return ppg, labels, subject_idx

# Define the AlexNet_1D Model
def AlexNet_1D(data_in_shape, num_output=2, kernel_size=3, useMaxPooling=True):
    """
    AlexNet-like 1D convolutional neural network for regression.

    Args:
        data_in_shape: Shape of the input data (ppg_length, 1).
        num_output: Number of outputs (default: 2 for SBP and DBP).
        kernel_size: Size of the convolution kernels.
        useMaxPooling: Whether to use max pooling after convolutional layers.

    Returns:
        model: Compiled Keras model.
    """
    # Define the input
    X_input = Input(shape=data_in_shape)

    # Convolutional stage
    X = Conv1D(96, 7, strides=3, padding="same", kernel_initializer=glorot_uniform(seed=0), name='conv1')(X_input)
    if useMaxPooling:
        X = MaxPooling1D(3, strides=2, name="MaxPool1")(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='BatchNorm1')(X)

    X = Conv1D(256, kernel_size=kernel_size, strides=1, padding="same", kernel_initializer=glorot_uniform(seed=0), name='conv2')(X)
    if useMaxPooling:
        X = MaxPooling1D(3, strides=2, name="MaxPool2")(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='BatchNorm2')(X)

    X = Conv1D(384, kernel_size=kernel_size, strides=1, padding="same", kernel_initializer=glorot_uniform(seed=0), name='conv3')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='BatchNorm3')(X)

    X = Conv1D(384, kernel_size=kernel_size, strides=1, padding="same", kernel_initializer=glorot_uniform(seed=0), name='conv4')(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='BatchNorm4')(X)

    X = Conv1D(256, kernel_size=kernel_size, strides=1, padding="same", kernel_initializer=glorot_uniform(seed=0), name='conv5')(X)
    if useMaxPooling:
        X = MaxPooling1D(3, strides=2, name="MaxPool5")(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='BatchNorm5')(X)

    # Fully connected stage
    X = Flatten()(X)
    X = Dense(4096, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='dense1')(X)
    X = Dropout(rate=0.5)(X)
    X = Dense(4096, activation='relu', kernel_initializer=glorot_uniform(seed=0), name='dense2')(X)
    X = Dropout(rate=0.5)(X)

    # Output stage
    X_SBP = Dense(1, activation='linear', kernel_initializer=glorot_uniform(seed=0), name='SBP')(X)
    X_DBP = Dense(1, activation='linear', kernel_initializer=glorot_uniform(seed=0), name='DBP')(X)

    # Create model
    model = Model(inputs=X_input, outputs=[X_SBP, X_DBP], name='AlexNet_1D')
    return model

# Main function to load data, preprocess, and train the model
def main():
    # Load data
    filename = "./Mimic.h5"
    ppg, labels, subject_idx = load_dataset(filename, num_samples=1000)  # Use only a subset for faster iteration
    
    # Reshape PPG to match input shape (batch_size, ppg_length, 1)
    ppg = ppg.reshape(ppg.shape[0], ppg.shape[1], 1)
    
    # Split labels into SBP and DBP
    sbp = labels[:, 0]
    dbp = labels[:, 1]

    # Define model
    model = AlexNet_1D(data_in_shape=(ppg.shape[1], 1))

    # Compile model
    model.compile(optimizer='adam', loss=['mse', 'mse'], metrics=['mae'])

    # Train model
    history = model.fit(
        ppg, [sbp, dbp],
        validation_split=0.2,
        epochs=10,
        batch_size=32
    )

    # Display model summary
    model.summary()

if __name__ == "__main__":
    main()
