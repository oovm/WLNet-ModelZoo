import tensorflow as tf
import wolframclient.serializers as wxf


def ckpt2wxf(path, name):
    ckpt = tf.train.get_checkpoint_state(path)
    reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
    all_variables = list(reader.get_variable_to_shape_map().keys())
    npy = dict(zip(all_variables, map(reader.get_tensor, all_variables)))
    wxf.export(npy, name + '.wxf', target_format='wxf')


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./checkpoints/color/deblur.model-523000.meta')
    saver.restore(sess, "./checkpoints/color/deblur.model-523000")
    saver.save(sess, "./tmp/color/model.ckpt")
    ckpt2wxf("./tmp/color/", "color")

# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('./checkpoints/lstm/deblur.model-523000.meta')
#     saver.restore(sess, "./checkpoints/lstm/deblur.model-523000")
#     saver.save(sess, "./tmp/lstm/model.ckpt")
#     ckpt2wxf("./tmp/lstm/", "lstm")
