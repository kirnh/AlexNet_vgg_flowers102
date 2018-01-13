import tensorflow as tf
import cv2
import numpy as np

image_path_list = ['/home/kh/Miscellaneous/vgg_flowers102/data/jpg/image_05325.jpg',
		   '/home/kh/Miscellaneous/vgg_flowers102/data/jpg/image_06246.jpg'] 
batch_size = 10

ckpt_dir = "/home/kh/Miscellaneous/WhoDat/_tmp/checkpoints"
meta_graph_file = "/home/kh/Miscellaneous/WhoDat/_tmp/checkpoints/model_epoch10.ckpt.meta"

def create_input_tensor(image_path_list, batch_size):
	input_tensor = []
	for image_path in image_path_list:
		image = cv2.imread(image_path)
		image = cv2.resize(image, (227, 227))
		input_tensor.append(image)
	dummy_image = np.zeros([227,227,3])
	while len(input_tensor)<batch_size:
		input_tensor.append(dummy_image)
	return input_tensor

latest_ckpt = tf.train.latest_checkpoint(ckpt_dir)

graph = tf.Graph()

with(graph.as_default()):
    sess = tf.Session()
    with(sess.as_default()):
        saver = tf.train.import_meta_graph(meta_graph_file)
        saver.restore(sess, latest_ckpt)

input_tensor = graph.get_operation_by_name('input').outputs[0]
keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
output = graph.get_operation_by_name('fc8/fc8').outputs[0]

in_tensor = create_input_tensor(image_path_list, batch_size)
logits = sess.run(output, feed_dict={input_tensor:in_tensor,
								keep_prob:1.0})

print("\nClass predictions:\n")

index = 0
for image_path in image_path_list:
	logit = logits[index]
	print("{} : {}".format(image_path, np.argmax(logit)))
	index+=1
