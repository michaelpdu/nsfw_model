import os
import time
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD
from pathlib import Path
from keras.applications import InceptionV3
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

def build_model(weights_file, shape):
    conv_base = InceptionV3(weights='imagenet', include_top=False, input_shape=shape)
    channel = shape[3]
    
    # First time run, no unlocking
    conv_base.trainable = False

    # Let's see it
    print('Summary')
    print(conv_base.summary())

    # Let's construct that top layer replacement
    x = conv_base.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x - Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_initializer=initializers.he_normal(seed=None), kernel_regularizer=regularizers.l2(.0005))(x)
    x = Dropout(0.5)(x)
    # Essential to have another layer for better accuracy
    x = Dense(128,activation='relu', kernel_initializer=initializers.he_normal(seed=None))(x)
    x = Dropout(0.25)(x)
    predictions = Dense(channel,  kernel_initializer="glorot_uniform", activation='softmax')(x)

    print('Stacking New Layers')
    model = Model(inputs = conv_base.input, outputs=predictions)

    # Load checkpoint if one is found
    if os.path.exists(weights_file):
        print ("loading ", weights_file)
        model.load_weights(weights_file)

    return model

def train_model(image_dir, nb_gpu):
    # No kruft plz
    clear_session()

    # Config
    height = constants.SIZES['basic']
    width = height
    channel = constants.NUM_CLASSES
    weights_file = "weights.best_inception_" + str(height) + '_gpu' + str(nb_gpu) + ".hdf5"

    if nb_gpu <= 1:
        print("[INFO] training with 1 GPU...")
        model = build_model(weights_file, shape=(height, width, channel))
    else:
        print("[INFO] training with {} GPUs...".format(nb_gpu))
    
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model = build_model(weights_file, shape=(height, width, channel))
        
        # make the model parallel
        model = multi_gpu_model(model, gpus=nb_gpu)

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
        workers=0,
        use_multiprocessing=True,
        validation_data=validation_generator,
        validation_steps=constants.VALIDATION_STEPS
    )
    print('Total time:', time.time()-start)

    # Save it for later
    print('Saving Model')
    # model.save("nsfw." + str(width) + "x" + str(height) + ".h5")
    model.save("nsfw." + str(width) + "x" + str(height) + '.gpu' + str(nb_gpu) + ".h5")

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
    plt.savefig('gpu_{}_lines.jpg'.format(nb_gpu))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=str, help="Path to image data, which includes train/test folders")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs")
    args = parser.parse_args()
    train_model(args.image_dir, args.gpus)