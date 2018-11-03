import tensorflow as tf
import wolframclient.serializers as wxf


def ckpt2wxf(path, name):
    ckpt = tf.train.get_checkpoint_state(path)
    reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
    all_variables = list(reader.get_variable_to_shape_map().keys())
    npy = dict(zip(all_variables, map(reader.get_tensor, all_variables)))
    wxf.export(npy, name + '.wxf', target_format='wxf')


'''
ckpt2wxf('./checkpoint/Fuji/', 'see fuji in the dark')
ckpt2wxf('./checkpoint/Sony/', 'see sony in the dark')
'''
