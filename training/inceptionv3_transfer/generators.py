import os
from keras.preprocessing.image import ImageDataGenerator
import constants

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data should not be modified
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

def create_generators(height, width, image_dir=constants.BASE_DIR, nb_gpu=1):
    print('Base dir:', image_dir)
    train_dir = os.path.join(image_dir, 'train')
    test_dir = os.path.join(image_dir, 'test')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=constants.GENERATOR_BATCH_SIZE * nb_gpu
    )

    validation_generator = validation_datagen.flow_from_directory(
        test_dir,
        target_size=(height, width),
        class_mode='categorical',
        batch_size=constants.GENERATOR_BATCH_SIZE * nb_gpu
    )

    return[train_generator, validation_generator]