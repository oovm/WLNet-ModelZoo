import os
import pickle

import numpy as np
import tensorflow as tf
import wolframclient.serializers as wxf

name = 'karras2018iclr-celebahq-1024x1024'
file = open(name + '.pkl', 'rb')
sess = tf.InteractiveSession()
G, D, Gs = pickle.load(file)
saver = tf.train.Saver()
save_path = "./tmp/" + name + "/"
model_name = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path_full = os.path.join(save_path, model_name)
saver.save(sess, save_path_full)

ckpt = tf.train.get_checkpoint_state(save_path)
reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
all_variables = list(reader.get_variable_to_shape_map().keys())
npy = dict(zip(all_variables, map(reader.get_tensor, all_variables)))
# remove `float32` because it had not be supported
npy.pop('D_paper/lod')
npy.pop('G_paper/lod')
npy.pop('G_paper_1/lod')

wxf.export(npy, name + '.wxf', target_format='wxf')

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:])  # 1000 random latents
latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10

wxf.export(latents, 'latents.wxf', target_format='wxf')
