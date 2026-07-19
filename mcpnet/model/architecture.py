"""
MCP-Net architecture and the ablation variants used for Table A (architecture
ablation) and Table C (multi-scale pyramid x decoder fusion factorial).

"Raw-input reinjection" and "cross-scale connections" are the SAME
mechanism: the `connections` argument threaded through `down_pyramid`. Only
the first pyramid stage ever touches raw pixels directly; every later stage
receives the previous stage's processed features and re-pools/concatenates
them forward via `connections`.

TABLE A / C — a 2x2 factorial isolating pyramid vs. connections independently
--------------------------------------------------------------------------
`down_pyramid` supports `nb_scales=1`, which — chained across depths with
`connections` threaded through exactly as before — gives a single-path
encoder that still has the connections mechanism
(`build_single_path_with_connections`). Combined with the other 3 variants:

                          | no connections            | with connections
    single-path encoder  | vanilla_unet               | single_path_with_connections
    multi-scale pyramid  | mcp_net_no_connections      | full_mcp_net

Reading down a column isolates the connections mechanism's effect; reading
across a row isolates the multi-scale pyramid's effect, independently of
each other.

TABLE D — decoder fusion ablation
--------------------------------------------------------------------------
`up_pyramid_additive` implements additive (Add) skip fusion in the decoder
as an alternative to the shipped concatenation-based `up_pyramid`, so the
fusion choice can be reported as an ablation rather than left implicit.
"""

from tensorflow.keras.layers import Add, Concatenate, Conv2D, Input, MaxPooling2D
from tensorflow.keras.models import Model

from mcpnet.model.layers import ActiveConv2D, UpSamplingBlock

DEFAULT_FILTERS = [16, 32, 64, 64, 64]  # p1..p5
DEFAULT_KERNELS = [3, 3, 3, 3, 3]


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
def input_pyramid(inputs, nb_scales):
    c = inputs
    outs = [c]
    for i in range(nb_scales - 1):
        c = MaxPooling2D(name=f"input_scale{i+1}")(c)
        outs.append(c)
    return outs


def down_pyramid(inputs, nb_scales, filters, kernels, nb_pyramid, groups=1, connections=None):
    """`connections=None` (the default) gives the "no cross-scale
    connections" behaviour used by `mcp_net_no_connections`.
    """
    convs = []
    conx = connections
    for i in range(nb_scales):
        c = inputs[i]
        if conx is not None:
            conx = MaxPooling2D(name=f"downsample_{i+1}_p{nb_pyramid}")(conx)
            conx = Concatenate(axis=-1, name=f"connection_{i+1}_dp_{nb_pyramid}")([conx, inputs[i]])
            c = conx
        c = ActiveConv2D(inputs=c, filters=filters, kernel=kernels)
        convs.append(c)
    return convs


def up_pyramid(inputs, connections, nb_scales, filters, kernels, nb_pyramid, groups=1):
    assert nb_scales == len(connections), f"Expected {nb_scales} connections, got {len(connections)}"
    assert len(filters) == nb_scales, f"Expected {nb_scales} filters, got {len(filters)}"
    assert len(kernels) == nb_scales, f"Expected {nb_scales} kernels, got {len(kernels)}"
    c = inputs
    convs = []
    for i in range(nb_scales):
        u = UpSamplingBlock(inputs=c, name=f"up_{i+1}_pyramid_{nb_pyramid}")
        u = Concatenate(axis=-1, name=f"connection_{i+1}_up_{nb_pyramid}")([u, connections[i]])
        c = ActiveConv2D(inputs=u, filters=filters[i], kernel=kernels[i], groups=groups)
        c = ActiveConv2D(inputs=c, filters=filters[i], kernel=kernels[i], groups=groups)
        convs.append(c)
    return convs


# --------------------------------------------------------------------------- #
# Variant 1: Full MCP-Net (the submitted model)
# --------------------------------------------------------------------------- #
def build_full_mcp_net(input_shape=(128, 128, 1), num_classes=4, name="full_mcp_net"):
    inputs = Input(input_shape)

    inp_pyramid = input_pyramid(inputs=inputs, nb_scales=5)
    p1 = down_pyramid(inputs=inp_pyramid, nb_scales=5, filters=16, kernels=3, nb_pyramid=1)
    p2 = down_pyramid(inputs=p1[1:], nb_scales=4, filters=32, kernels=3, nb_pyramid=2, connections=p1[0])
    p3 = down_pyramid(inputs=p2[1:], nb_scales=3, filters=64, kernels=3, nb_pyramid=3, connections=p2[0])
    p4 = down_pyramid(inputs=p3[1:], nb_scales=2, filters=64, kernels=3, nb_pyramid=4, connections=p3[0])
    p5 = down_pyramid(inputs=p4[1:], nb_scales=1, filters=64, kernels=3, nb_pyramid=5, connections=p4[0])

    p6 = up_pyramid(
        inputs=p5[0],
        connections=[p4[0], p3[0], p2[0], p1[0]],
        nb_scales=4,
        filters=[64, 64, 32, 16],
        kernels=[3, 3, 3, 3],
        nb_pyramid=6,
    )

    outputs = ActiveConv2D(
        inputs=p6[-1], kernel=1, filters=num_classes, activation="softmax", name="predictions", batch_norm=False
    )
    return Model(inputs=inputs, outputs=outputs, name=name)


# --------------------------------------------------------------------------- #
# Variant 2: MCP-Net without cross-scale connections
# (multi-scale parallel branches kept, but `connections` never threaded)
# --------------------------------------------------------------------------- #
def build_mcp_net_no_connections(input_shape=(128, 128, 1), num_classes=4, name="mcp_net_no_connections"):
    inputs = Input(input_shape)

    inp_pyramid = input_pyramid(inputs=inputs, nb_scales=5)
    p1 = down_pyramid(inputs=inp_pyramid, nb_scales=5, filters=16, kernels=3, nb_pyramid=1)  # connections=None
    p2 = down_pyramid(inputs=p1[1:], nb_scales=4, filters=32, kernels=3, nb_pyramid=2)  # connections=None
    p3 = down_pyramid(inputs=p2[1:], nb_scales=3, filters=64, kernels=3, nb_pyramid=3)
    p4 = down_pyramid(inputs=p3[1:], nb_scales=2, filters=64, kernels=3, nb_pyramid=4)
    p5 = down_pyramid(inputs=p4[1:], nb_scales=1, filters=64, kernels=3, nb_pyramid=5)

    # up_pyramid still needs 4 skip tensors of matching resolution; without the
    # connections mechanism, p1[0]/p2[0]/p3[0]/p4[0] are just the plain
    # per-scale conv outputs (no forwarded high-res info baked in) — this is
    # exactly the point of the ablation.
    p6 = up_pyramid(
        inputs=p5[0],
        connections=[p4[0], p3[0], p2[0], p1[0]],
        nb_scales=4,
        filters=[64, 64, 32, 16],
        kernels=[3, 3, 3, 3],
        nb_pyramid=6,
    )

    outputs = ActiveConv2D(
        inputs=p6[-1], kernel=1, filters=num_classes, activation="softmax", name="predictions", batch_norm=False
    )
    return Model(inputs=inputs, outputs=outputs, name=name)


# --------------------------------------------------------------------------- #
# Variant 3: Vanilla U-Net anchor (single path, standard skip connections)
# Matches depth (5 levels) and filter progression (16,32,64,64,64) for a fair
# parameter-count-adjacent comparison.
# --------------------------------------------------------------------------- #
def build_vanilla_unet(input_shape=(128, 128, 1), num_classes=4, name="vanilla_unet"):
    inputs = Input(input_shape)
    filters = DEFAULT_FILTERS
    kernels = DEFAULT_KERNELS

    skips = []
    x = inputs
    for depth in range(5):
        x = ActiveConv2D(inputs=x, filters=filters[depth], kernel=kernels[depth], name=f"enc_conv_{depth}")
        if depth < 4:
            skips.append(x)
            x = MaxPooling2D(name=f"enc_pool_{depth}")(x)

    # x is now the bottleneck (1/16 resolution, matching the pyramid's p5)
    up_filters = [64, 64, 32, 16]
    for depth in range(4):
        x = UpSamplingBlock(inputs=x, name=f"dec_up_{depth}")
        x = Concatenate(axis=-1, name=f"dec_concat_{depth}")([x, skips[3 - depth]])
        x = ActiveConv2D(inputs=x, filters=up_filters[depth], kernel=3, name=f"dec_conv1_{depth}")
        x = ActiveConv2D(inputs=x, filters=up_filters[depth], kernel=3, name=f"dec_conv2_{depth}")

    outputs = ActiveConv2D(
        inputs=x, kernel=1, filters=num_classes, activation="softmax", name="predictions", batch_norm=False
    )
    return Model(inputs=inputs, outputs=outputs, name=name)


# --------------------------------------------------------------------------- #
# Variant 4: Single-path encoder WITH connections (isolates the pyramid
# effect independently of the connections effect — completes the 2x2)
# --------------------------------------------------------------------------- #
def build_single_path_with_connections(input_shape=(128, 128, 1), num_classes=4, name="single_path_with_connections"):
    """Same connections mechanism as full_mcp_net, but the encoder tracks
    only ONE scale per depth (nb_scales=1 throughout) instead of a shrinking
    multi-scale branch list.
    """
    inputs = Input(input_shape)
    x = inputs
    conx = None
    skips = []

    for depth in range(5):
        out = down_pyramid(
            inputs=[x], nb_scales=1, filters=DEFAULT_FILTERS[depth], kernels=3,
            nb_pyramid=depth + 1, connections=conx,
        )
        x = out[0]
        skips.append(x)
        conx = x  # thread this depth's single-scale output forward as next depth's connection
        if depth < 4:
            x = MaxPooling2D(name=f"single_path_pool_{depth}")(x)

    p6 = up_pyramid(
        inputs=skips[4],
        connections=[skips[3], skips[2], skips[1], skips[0]],
        nb_scales=4,
        filters=[64, 64, 32, 16],
        kernels=[3, 3, 3, 3],
        nb_pyramid=6,
    )

    outputs = ActiveConv2D(
        inputs=p6[-1], kernel=1, filters=num_classes, activation="softmax", name="predictions", batch_norm=False
    )
    return Model(inputs=inputs, outputs=outputs, name=name)


# --------------------------------------------------------------------------- #
# Decoder fusion variants
# --------------------------------------------------------------------------- #
def up_pyramid_additive(inputs, connections, nb_scales, filters, kernels, nb_pyramid, groups=1):
    """Same as up_pyramid, but fuses the skip connection via Add instead of
    Concatenate. Add requires matching channel counts, so a 1x1 conv projects
    the skip connection to match the upsampled tensor's channel count before
    summing.
    """
    assert nb_scales == len(connections), f"Expected {nb_scales} connections, got {len(connections)}"
    assert len(filters) == nb_scales, f"Expected {nb_scales} filters, got {len(filters)}"
    assert len(kernels) == nb_scales, f"Expected {nb_scales} kernels, got {len(kernels)}"
    c = inputs
    convs = []
    for i in range(nb_scales):
        u = UpSamplingBlock(inputs=c, name=f"up_{i+1}_pyramid_{nb_pyramid}_add")
        skip = connections[i]
        # project skip connection to match u's channel count for Add
        target_channels = u.shape[-1]
        if skip.shape[-1] != target_channels:
            skip = Conv2D(target_channels, kernel_size=1, padding="same",
                          name=f"skip_proj_{i+1}_pyramid_{nb_pyramid}")(skip)
        fused = Add(name=f"connection_{i+1}_up_{nb_pyramid}_add")([u, skip])
        c = ActiveConv2D(inputs=fused, filters=filters[i], kernel=kernels[i], groups=groups)
        c = ActiveConv2D(inputs=c, filters=filters[i], kernel=kernels[i], groups=groups)
        convs.append(c)
    return convs


def build_full_mcp_net_additive_decoder(input_shape=(128, 128, 1), num_classes=4, name="full_mcp_net_additive_decoder"):
    """Identical encoder to full_mcp_net; decoder uses additive fusion
    (up_pyramid_additive) instead of concatenation. Holds everything else
    fixed so this isolates the decoder fusion choice specifically.
    """
    inputs = Input(input_shape)

    inp_pyramid = input_pyramid(inputs=inputs, nb_scales=5)
    p1 = down_pyramid(inputs=inp_pyramid, nb_scales=5, filters=16, kernels=3, nb_pyramid=1)
    p2 = down_pyramid(inputs=p1[1:], nb_scales=4, filters=32, kernels=3, nb_pyramid=2, connections=p1[0])
    p3 = down_pyramid(inputs=p2[1:], nb_scales=3, filters=64, kernels=3, nb_pyramid=3, connections=p2[0])
    p4 = down_pyramid(inputs=p3[1:], nb_scales=2, filters=64, kernels=3, nb_pyramid=4, connections=p3[0])
    p5 = down_pyramid(inputs=p4[1:], nb_scales=1, filters=64, kernels=3, nb_pyramid=5, connections=p4[0])

    p6 = up_pyramid_additive(
        inputs=p5[0],
        connections=[p4[0], p3[0], p2[0], p1[0]],
        nb_scales=4,
        filters=[64, 64, 32, 16],
        kernels=[3, 3, 3, 3],
        nb_pyramid=6,
    )

    outputs = ActiveConv2D(
        inputs=p6[-1], kernel=1, filters=num_classes, activation="softmax", name="predictions", batch_norm=False
    )
    return Model(inputs=inputs, outputs=outputs, name=name)


# --------------------------------------------------------------------------- #
# Factory used by the ablation runner script
# --------------------------------------------------------------------------- #
VARIANT_BUILDERS = {
    "full_mcp_net": build_full_mcp_net,
    "mcp_net_no_connections": build_mcp_net_no_connections,
    "vanilla_unet": build_vanilla_unet,
    "single_path_with_connections": build_single_path_with_connections,       # Table A/C, 2x2 factorial
    "full_mcp_net_additive_decoder": build_full_mcp_net_additive_decoder,     # Table D, decoder fusion
}


def get_model(variant_name, input_shape=(128, 128, 1), num_classes=4):
    if variant_name not in VARIANT_BUILDERS:
        raise ValueError(f"Unknown variant '{variant_name}'. Options: {list(VARIANT_BUILDERS)}")
    return VARIANT_BUILDERS[variant_name](input_shape=input_shape, num_classes=num_classes, name=variant_name)
