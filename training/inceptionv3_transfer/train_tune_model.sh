IMAGE_DIR="./yahoo_nsfw_data2/"
python train_initialization_multi_gpus.py -g=1 $IMAGE_DIR
python train_fine_tune_multi_gpus.py -g=1 -m=./nsfw.299x299.gpu1.h5 $IMAGE_DIR