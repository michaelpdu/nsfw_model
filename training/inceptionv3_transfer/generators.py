import os
from keras.preprocessing.image import ImageDataGenerator
from keras_applications.imagenet_utils import preprocess_input
import constants

def preprocessing_function_caffe(x):
    return preprocess_input(x, data_format='channels_last', mode='caffe')

def preprocessing_function_tf(x):
    return preprocess_input(x, data_format='channels_last', mode='tf')

def create_generators(height, width, image_dir=constants.BASE_DIR, mode='caffe', nb_gpu=1):
    print('Base dir:', image_dir)
    train_dir = os.path.join(image_dir, 'train')
    test_dir = os.path.join(image_dir, 'test')

    if mode == 'caffe':
        preprocessing = preprocessing_function_caffe
    else:
        preprocessing = preprocessing_function_tf

    train_datagen = ImageDataGenerator(
        # rescale=1./255,
        preprocessing_function=preprocessing,
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
        # rescale=1./255
        preprocessing_function=preprocessing
    )

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