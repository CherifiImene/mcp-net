from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Dropout
)
from tensorflow.keras.models import Model


def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return MaxPooling2D()(x)


def build_localizer(input_shape=(256, 256, 1)):
    # GlobalAveragePooling before the dense head means this architecture
    # doesn't actually care what H,W the input is -- only used here with a
    # fixed input_shape because Keras needs a concrete shape to build the
    # graph, but the same trained weights would work on other sizes too
    # via a functional-API rebuild if ever needed
    inputs = Input(input_shape)
    x = conv_block(inputs, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    # sigmoid keeps output naturally bounded to [0,1], matching the
    # normalized bbox target convention
    outputs = Dense(4, activation="sigmoid", name="bbox")(x)

    return Model(inputs, outputs, name="heart_localizer")