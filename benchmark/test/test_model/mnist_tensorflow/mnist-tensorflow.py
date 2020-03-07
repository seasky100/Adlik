"""
CNN-MNIST
"""
import tensorflow as tf
import os


def load_data():
    mnist_dir = os.path.join(os.path.dirname(__file__), 'MNIST_data/')
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets(mnist_dir, one_hot=True)


def conv_layer(x_data, shape, init_value, strides, padding):
    weight_conv = tf.Variable(tf.truncated_normal(shape=shape, stddev=init_value[0]))
    bias_conv = tf.Variable(tf.constant(value=init_value[1], shape=[shape[-1]]))
    conv = tf.nn.relu(tf.nn.conv2d(x_data, weight_conv, strides=strides, padding=padding) + bias_conv)

    return conv


def data_flatten(input_fc):
    shape = input_fc.shape
    input_flatten = tf.reshape(input_fc, [-1, shape[1] * shape[2] * shape[3]])
    return input_flatten


def fully_connected(fc_data, units, std, value):
    fc_shape = fc_data.get_shape().as_list()
    weight_fc = tf.Variable(tf.truncated_normal(shape=[fc_shape[1], units], stddev=std))
    bias_fc = tf.Variable(tf.constant(value=value, shape=[units]))
    return tf.matmul(fc_data, weight_fc) + bias_fc


def output_cnn(images):
    images = tf.reshape(images, [-1, 28, 28, 1])
    print(images.name)
    # conv1 layer
    conv1 = conv_layer(images, shape=[5, 5, 1, 32], init_value=[0.1, 0.1], strides=[1, 1, 1, 1], padding='SAME')

    # pooling layer1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv2 layer
    conv2 = conv_layer(pool1, shape=[5, 5, 32, 64], init_value=[0.1, 0.1], strides=[1, 1, 1, 1], padding='SAME')

    # pooling layer2
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fully connected layer
    pool2_flatten = data_flatten(pool2)
    fc1 = tf.nn.relu(fully_connected(pool2_flatten, 1024, 0.1, 0.1))

    # output layer
    fc1_drop = tf.nn.dropout(fc1, rate=0.5)

    y_cnn = tf.nn.softmax(fully_connected(fc1_drop, 10, 0.1, 0.1))

    return y_cnn


def get_loss(y_predict, y_label):
    return -tf.reduce_sum(y_label * tf.log(y_predict))


def get_optimizer(optimizer, learn_rate):
    return optimizer(learn_rate)


def accuracy(y_predict, y_label):
    correct_predict = tf.equal(tf.math.argmax(y_predict, 1), tf.math.argmax(y_label, 1))
    return tf.reduce_mean(tf.cast(correct_predict, 'float'))


def main():
    """
    train cnn model
    """
    x_data = tf.placeholder("float", [None, 28 * 28])
    y_label = tf.placeholder("float", [None, 10])
    y_predict = output_cnn(x_data)
    loss = get_loss(y_predict, y_label)
    optimizer = get_optimizer(tf.train.AdamOptimizer, learn_rate=1e-4)
    train_op = optimizer.minimize(loss)
    class_accuracy = accuracy(y_predict, y_label)
    train_data = load_data()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            train_images, train_label = train_data.train.next_batch(50)
            _, train_accuracy = sess.run([train_op, class_accuracy],
                                         feed_dict={x_data: train_images, y_label: train_label})
            if i % 100 == 0:
                print(f'step: {i}, accuracy {train_accuracy}')
        save_path = os.path.join(os.path.dirname(__file__), 'model', 'mnist.ckpt')
        dir_name = os.path.dirname(save_path)
        os.makedirs(dir_name, exist_ok=True)
        tf.train.Saver().save(sess, save_path)


if __name__ == '__main__':
    main()
