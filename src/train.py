from keras.preprocessing.image import ImageDataGenerator


def train_on_gen(dirname, shape_target=(224,224)):
    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        directory=dirname,
        target_size=shape_target,
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    return train_generator