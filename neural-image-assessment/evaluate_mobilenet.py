import numpy as np
import argparse
from path import Path
import os

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from PIL import Image

from utils.score_utils import mean_score, std_score

parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it')

parser.add_argument('-img', type=str, default=[None], nargs='+',
                    help='Pass one or more image paths to evaluate them')

parser.add_argument('-resize', type=str, default='false',
                    help='Resize images to 224x224 before scoring')

parser.add_argument('-rank', type=str, default='true',
                    help='Whether to tank the images after they have been scored')

args = parser.parse_args()
resize_image = args.resize.lower() in ("true", "yes", "t", "1")
target_size = (224, 224) if resize_image else None
rank_images = args.rank.lower() in ("true", "yes", "t", "1")


def get_image_paths(directory):
    image_paths = {}
    diff_ = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                subdir_ = os.path.relpath(root, directory)
                if subdir_ not in image_paths:
                    image_paths[subdir_] = []
                image_paths[subdir_].append(os.path.join(root, file))
            if file.endswith('.txt'):
                subdir_ = os.path.relpath(root, directory)
                if subdir_ not in diff_:
                    diff_[subdir_] = []
                diff_[subdir_].append(os.path.join(root, file))
    return image_paths, diff_


# give priority to directory
if args.dir is not None:
    print("Loading images from directory : ", args.dir)
    imgs, diff = get_image_paths(args.dir)

else:
    raise RuntimeError('Either -dir or -img arguments must be passed as argument')

with tf.device('/CPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/mobilenet_weights.h5')

    score_list = []

    for subdir, images in imgs.items():
        subdir_score_list = []
        diff_path = diff[subdir][0]
        # 这里需要进行重新采样
        diff_data = []
        saved_num = []
        re_images = []
        with open(diff_path, "r") as file:
            cnt = 1
            for line in file:
                val = float(line.strip())
                diff_data.append((cnt, val))
                cnt += 1

        diff_data = sorted(diff_data, key=lambda x: x[1], reverse=True)
        num = int(len(diff_data) * 0.6)

        diff_data = diff_data[0: num + 1]
        for save_num , _ in diff_data:
            saved_num.append(save_num)
            saved_num.append(save_num+1)
        saved_num = list(set(saved_num))

        for img in images:
            seq = img[-8:-4]
            seq = int(seq)
            if seq in saved_num:
                re_images.append(img)

        for img in re_images:
            img = load_img(img, target_size=target_size)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            x = preprocess_input(x)

            scores = model.predict(x, batch_size=1, verbose=0)[0]

            mean = mean_score(scores)
            std = std_score(scores)

            file_name = Path(images).name.lower()
            subdir_score_list.append((file_name, mean))
        subdir_score_sum = 0.0
        for score in subdir_score_list:
            subdir_score_sum += score[1]
        score_list.append((subdir, subdir_score_sum))
    #            print("Evaluating : ", img_path)
    #            print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
    #            print()

    score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(score_list):
        print(f"{name}:  score = {score}")
        break
#   if rank_images:
#      print("*" * 40, "Ranking Images", "*" * 40)
#       score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
#
#       for i, (name, score) in enumerate(score_list):
#            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))
