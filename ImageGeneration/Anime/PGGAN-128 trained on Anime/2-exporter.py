import os
import pickle
import tensorflow as tf
import wolframclient.serializers as wxf

name = 'network-snapshot-057891'
out_name = 'PGGAN-128 trained on Anime'

file = open(name + '.pkl', 'rb')
sess = tf.InteractiveSession()
G, D, Gs = pickle.load(file)
saver = tf.train.Saver()
save_path = "./target/" + name + "/"
model_name = 'model'
if not os.path.exists(save_path):
    os.makedirs(save_path)
save_path_full = os.path.join(save_path, model_name)
saver.save(sess, save_path_full)

ckpt = tf.train.get_checkpoint_state(save_path)
reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
all_variables = list(reader.get_variable_to_shape_map().keys())
npy = dict(zip(all_variables, map(reader.get_tensor, all_variables)))
wxf.export(npy, out_name + '.WXF', target_format='wxf')

# Save as protobuf
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=['Gs/images_out']
        # output_node_names=['Gs/ToRGB_lod0/add']
    )

    with tf.gfile.GFile("./target/" + name + ".pb", "wb") as file:  # 保存模型
        file.write(output_graph_def.SerializeToString())  # 序列化输出
