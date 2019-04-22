import os
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.backend import clear_session
from keras.optimizers import SGD
from pathlib import Path
from keras.models import Sequential, Model, load_model
from keras.utils import multi_gpu_model

# reusable stuff
import constants
import callbacks
import generators

def init_model(model_file, weights_file):
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

    # Load checkpoint if one is found
    if os.path.exists(weights_file):
            print ("loading ", weights_file)
            model.load_weights(weights_file)
    return model

def fine_tune_model(nb_gpu):
    # No kruft plz
    clear_session()

    # Config
    height = constants.SIZES['basic']
    width = height
    model_file = "nsfw." + str(width) + "x" + str(height) + ".h5"
    weights_file = "weights.best_inception" + str(height) + ".hdf5"

    if nb_gpu <= 1:
        print("[INFO] training with 1 GPU...")
        model = init_model(model_file, weights_file)
    else:
        print("[INFO] training with {} GPUs...".format(nb_gpu))
    
        # we'll store a copy of the model on *every* GPU and then combine
        # the results from the gradient updates on the CPU
        with tf.device("/cpu:0"):
            # initialize the model
            model = init_model(model_file, weights_file)
        
        # make the model parallel
        model = multi_gpu_model(model, gpus=nb_gpu)

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
    train_generator, validation_generator = generators.create_generators(height, width, nb_gpu)

    print('Start training!')
    history = model.fit_generator(
        train_generator,
        callbacks=callbacks_list,
        epochs=constants.TOTAL_EPOCHS,
        steps_per_epoch=constants.STEPS_PER_EPOCH,
        shuffle=True,
        # having crazy threading issues
        # set workers to zero if you see an error like: 
        # `freeze_support()`
        workers=0,
        use_multiprocessing=True,
        validation_data=validation_generator,
        validation_steps=constants.VALIDATION_STEPS
    )

    # Save it for later
    print('Saving Model')
    model.save("nsfw." + str(width) + "x" + str(height) + ".h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpus", type=int, defalut=1, help="Number of GPUs")
    args = parser.parse_args()
    fine_tune_model(args.gpus)