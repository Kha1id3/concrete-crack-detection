from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.optimizers import Adam


def build_mobilenetv2_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
) -> Model:
    """Build a simple MobileNetV2 transfer learning model."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze the pretrained feature extractor for the first baseline.
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_tuned_mobilenetv2_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
    fine_tune_layers: int = 5,
    use_augmentation: bool = False,
) -> Model:
    """Build a MobileNetV2 model with the last layers unfrozen for fine-tuning."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Start by freezing everything, then unfreeze only the last few layers.
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base_model.layers[-fine_tune_layers:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    inputs = Input(shape=input_shape)
    x = inputs

    # Keras augmentation layers are active during training only.
    if use_augmentation:
        x = RandomFlip("horizontal")(x)
        x = RandomRotation(0.05)(x)
        x = RandomZoom(0.1)(x)

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
