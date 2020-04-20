import re
import os
import numpy as np
import tensorflow as tf
import json

from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import Progbar
from keras.models import model_from_json
import utils
import img_utils
from common_flags import FLAGS

test_datagen = utils.DroneDataGenerator(rescale=1./255)
y = img_utils.load_img("data/testing/HMB_3/images/1479425597710017397.jpg",
                    grayscale=True,
                    crop_size=(200,200),
                    target_size=(320,240))

x = test_datagen.random_transform(y,seed=None)
x = test_datagen.standardize(x)
batch_x = np.zeros((1,) + (200,200,1),
                dtype=K.floatx())
batch_x[0]=x
json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
model = utils.jsonToModel(json_model_path)

    # Load weights
weights_load_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.weights_fname)
try:
    model.load_weights(weights_load_path)
    print("Loaded model from {}".format(weights_load_path))
except:
    print("Impossible to find weight path. Returning untrained model")
outs = model.predict(batch_x)
print(outs)
