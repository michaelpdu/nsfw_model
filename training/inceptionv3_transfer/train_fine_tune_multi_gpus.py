import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD
from pathlib import Path
from keras.models import Sequential, Model, load_model
from keras.layers import Dense
from keras.utils import multi_gpu_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# reusable stuff
import constants
import callbacks
import generators

def init_model(model_file):
    print ('Starting from last full model run')
    model = load_model(model_file)

    # Unlock a few layers deep in Inception v3
    model.trainable = False
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'conv2d_56':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # Let's see it
    print('Summary')
    print(model.summary())

    return model

def fine_tune_model(model_file, image_dir, nb_gpu):
    # No kruft plz
    clear_session()

    # Config
    height = constants.SIZES['basic']
    width = height
    # model_file = "nsfw." + str(width) + "x" + str(height) + ".h5"
    weights_file = "weights.best_inception_" + str(height) + '_gpu' + str(nb_gpu) + ".h5"

    if nb_gpu <= 1:
        print("[INFO] training with 1 GPU...")
        model = init_model(model_file)
    else:
        print("[INFO] training with {} GPUs...".format(nb_gpu))
    
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model = init_model(model_file)
        
        # make the model parallel
        model = multi_gpu_model(model, gpus=nb_gpu)

        # # add dense layer for merged model
        # from keras import backend as K
        # x = model.output
        # predictions = K.bias_add(x, 0)
        # # predictions = Dense(2)(x)
        # model = Model(inputs = model.input, outputs=predictions)

    # Get all model callbacks
    callbacks_list = callbacks.make_callbacks(weights_file)

    print('Compile model')
    opt = SGD(momentum=.9)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    # Get training/validation data via generators
    train_generator, validation_generator = generators.create_generators(\
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
        workers=96,
        use_multiprocessing=True,
        validation_data=validation_generator,
        validation_steps=constants.VALIDATION_STEPS
    )
    print('Total time:', time.time()-start)

    # Save it for later
    print('Saving Model')
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
    parser.add_argument("-m", "--model_file", type=str, help="path to initial model file")
    parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs")
    args = parser.parse_args()
    fine_tune_model(args.model_file, args.image_dir, args.gpus)