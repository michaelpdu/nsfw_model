import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD
from pathlib import Path
from keras.applications import InceptionV3, InceptionResNetV2, ResNet50
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras import initializers, regularizers
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# reusable stuff
import constants
import callbacks
import generators

import multiprocessing
NUM_CPU = multiprocessing.cpu_count()

def build_model(weights_file, type='inception_v3', shape=(299,299,3), nb_output=5):
    print('shape:', shape)
    if type == 'inception_v3':
        basic_model = InceptionV3(weights='imagenet', include_top=False, input_shape=shape)
    elif type == 'inception_resnet_v2':
        basic_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=shape)
    elif type == 'resnet50':
        basic_model = ResNet50(weights='imagenet', include_top=False, input_shape=shape)
    else:
        raise Exception('Unsupported Model Type!')

    # First time run, no unlocking
    basic_model.trainable = False

    # Let's see it
    print('Summary')
    print(basic_model.summary())

    # Let's construct that top layer replacement
    x = basic_model.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x - Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_initializer=initializers.he_normal(seed=None), kernel_regularizer=regularizers.l2(.0005))(x)
    x = Dropout(0.5)(x)
    # Essential to have another layer for better accuracy
    x = Dense(128,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
    x = Dropout(0.25)(x)
    predictions = Dense(nb_output,  kernel_initializer="glorot_uniform", activation='softmax')(x)

    print('Stacking New Layers')
    model = Model(inputs = basic_model.input, outputs=predictions)

    # Load checkpoint if one is found
    if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

    return model

def train_model(model_type, weights_file, image_dir, nb_gpu):
    # No kruft plz
    clear_session()

    # Config
    height = constants.SIZES['basic']
    width = height
    nb_classes = constants.NUM_CLASSES

    if nb_gpu <= 1:
        print("[INFO] training with 1 GPU...")
        model = build_model(weights_file, type=model_type, shape=(height, width, 3), nb_output=nb_classes)
    else:
        print("[INFO] training with {} GPUs...".format(nb_gpu))
    
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model = build_model(weights_file, type=model_type, shape=(height, width, 3), nb_output=nb_classes)
        
        # make the model parallel
        model = multi_gpu_model(model, gpus=nb_gpu)

        x = model.output
        x = Flatten()(x)
        predictions = Dense(nb_output, activation='softmax', name='new_outputs')(x)
        model = Model(inputs = model.input, outputs=predictions)

    if not os.path.exists(weights_file):
        weights_file = "weights.{}.{}.gpu{}.hdf5".format(model_type, height, nb_gpu)

    # Get all model callbacks
    callbacks_list = callbacks.make_callbacks(weights_file)

    print('Compile model')
    # originally adam, but research says SGD with scheduler
    # opt = Adam(lr=0.001, amsgrad=True)
    opt = SGD(momentum=.9)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # Get training/validation data via generators
    train_generator, validation_generator = generators.create_generators( \
        height, width, image_dir=image_dir, nb_gpu=nb_gpu)

    print('Start training!')
    start = time.time()
    history = model.fit_generator(
        train_generator,
        callbacks=callbacks_list,
        epochs=constants.TOTAL_EPOCHS,
        # steps_per_epoch=constants.STEPS_PER_EPOCH,
        steps_per_epoch=train_generator.samples//(constants.GENERATOR_BATCH_SIZE*nb_gpu),
        shuffle=True,
        # having crazy threading issues
        # set workers to zero if you see an error like: 
        # `freeze_support()`
        max_queue_size=100,
        workers=NUM_CPU,
        use_multiprocessing=True,
        validation_data=validation_generator,
        validation_steps=constants.VALIDATION_STEPS
    )
    print('Total time:', time.time()-start)

    # Save it for later
    print('Saving Model ...')
    model.save("nsfw.{}x{}.{}.gpu{}.h5".format(width, height, model_type, nb_gpu))

    # grab the history object dictionary
    H = history.history
    
    # plot the training loss and accuracy
    N = np.arange(0, len(H["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H["loss"], label="train_loss")
    plt.plot(N, H["val_loss"], label="test_loss")
    plt.plot(N, H["acc"], label="train_acc")
    plt.plot(N, H["val_acc"], label="test_acc")
    plt.title("Inception Model on NSFW Data")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    
    # save the figure
    plt.savefig('gpu_{}_train_lines.jpg'.format(nb_gpu))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, \
        help="Path to image data, which includes train/test folders")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("-w", "--weights_file", type=str, default='', help="Path to weights_file")
    parser.add_argument("-t", "--type", type=str, default='inception_v3', \
        help="Model type, inception_v3|inception_resnet_v2|resnet50|...")
    args = parser.parse_args()
    train_model(args.type, args.weights_file, args.image_dir, args.gpus)