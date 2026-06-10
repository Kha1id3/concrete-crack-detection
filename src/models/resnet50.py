from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Converts [0, 255] RGB inputs to BGR and subtracts ImageNet channel means.
PREPROCESS_FN = preprocess_input


def build_resnet50_model(
    input_shape: tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 1,
) -> Model:
    """Build a simple ResNet50 baseline transfer learning model."""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
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
