import tensorflow as tf


class model:
    def __init__(self, input_size):
        self.input_placeholder = tf.placeholder(tf.float32, (None, input_size))

        momentum = 0.90
        initializer = tf.contrib.layers.xavier_initializer()
        layer = self.input_placeholder
        layer_sizes = [100, 100, 100, 100]
        for i, size in enumerate(layer_sizes):
            layer = tf.nn.relu(tf.layers.dense(layer, size, kernel_initializer=initializer))
            layer = tf.layers.batch_normalization(layer, momentum=momentum)
            layer = tf.nn.dropout(layer, 0.5)

        layer = tf.nn.relu(tf.layers.dense(layer, 2, kernel_initializer=initializer))
        self.logits = tf.nn.softmax(layer)

        self.label_placeholder = tf.placeholder(tf.float32, (None, 2))
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.logits))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer().minimize(self.loss)
