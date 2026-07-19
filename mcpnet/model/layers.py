"""
Shared building blocks for MCP-Net and its ablation variants.

Kept identical across all variants so that architectural ablations (Table A)
differ ONLY in connectivity, not in the underlying conv block behaviour.
"""

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, UpSampling2D


def ActiveConv2D(
    inputs,
    filters,
    kernel,
    initializer="he_normal",
    name=None,
    dilation=1,
    activation="relu",
    groups=1,
    batch_norm=True,
    activation_first=False,
):
    """Conv2D -> (Activation) -> (BatchNorm).

    Changing this ordering to Conv -> BatchNorm -> Activation changes
    training dynamics for every model in this repo — treat it as an
    explicit ablation, not a drop-in fix.
    """
    assert (activation_first and activation) or not activation_first, (
        f"You specified activation first but no activation was given: {activation}"
    )

    tmp = inputs
    if activation_first:
        tmp = Activation(activation)(tmp)

    c = Conv2D(
        filters=filters,
        kernel_size=kernel,
        kernel_initializer=initializer,
        dilation_rate=dilation,
        padding="same",
        name=name,
        groups=groups,
    )(inputs)

    if not activation_first and activation:
        c = Activation(activation)(c)
    if batch_norm:
        c = BatchNormalization()(c)
    return c


def UpSamplingBlock(inputs, name, kernel=2, size=2, interpolation="nearest", filters="same"):
    if filters == "same":
        filters = K.int_shape(inputs)[3]
    else:
        assert isinstance(filters, int), (
            f"filters can be either a str 'same' or an int. Got filters : {type(filters)}"
        )

    up = UpSampling2D(size=size, interpolation=interpolation, name=f"upsampling_{name}")(inputs)
    up = ActiveConv2D(inputs=up, filters=filters, kernel=kernel, name=f"up_conv1_{name}")
    return up
