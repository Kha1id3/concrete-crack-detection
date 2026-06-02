from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


def build_efficientnetb0_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
) -> Model:
    """Build a simple EfficientNet-B0 baseline transfer learning model."""
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze the pretrained feature extractor for the baseline experiment.
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
