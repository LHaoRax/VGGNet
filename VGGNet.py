from datetime import datetime
import tensorflow as tf
import math
import time

num_batches = 100
batch_size = 32

# ============================================= Part1: 定义不同功能层的OP =================================================


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"W", shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, [1, dh, dw, 1], padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"W", shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name="b")
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]

    return activation


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

# =========================================== Part2: 定义整个网络层的结构 ==================================================


def inference_op(input_op, keep_prob):
    # the first conv layer
    p = []
    conv1_1 = conv_op(input_op, name='con1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name='con1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dh=2, dw=2)

    # the second conv layer
    conv2_1 = conv_op(pool1, name='con2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name='con2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dh=2, dw=2)

    # the third conv layer
    conv3_1 = conv_op(pool2, name='con3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='con3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='con3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)

    # the fourth conv layer
    conv4_1 = conv_op(pool3, name='con4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='con4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='con4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dh=2, dw=2)

    # the fifth conv layer
    conv5_1 = conv_op(pool4, name='con5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='con5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='con5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dh=2, dw=2)

    # flatten the conv layer
    shape = pool5.get_shape()
    shape_size = shape[1].value * shape[2].value * shape[3].value
    resh1 = tf.reshape(pool5, [-1, shape_size], name="resh1")

    # the sixth fc layer
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    # the seventh fc layer
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    # the last layer
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p

# =============================================== Part3: 定义时间计算功能 =================================================


def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        end_time = time.time()
        duration = end_time - start_time

        if i >= num_steps_burn_in:
            if i % 10 == 0:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration ** 2

    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn ** 2
    sd = math.sqrt(vr)

    print('%s: %s across %d steps, %.3f +/- %.3f sec/batches' %
          (datetime.now(), info_string, num_batches, mn, sd))

# ============================================== Part4: 定义主要入口函数 ==================================================


def run_banch():
    with tf.Graph().as_default():
        image_size = 224
        image = tf.Variable(tf.random_normal([batch_size,
                                              image_size,
                                              image_size, 3],
                                             dtype=tf.float32,
                                             stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(image, keep_prob)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Backward")

run_banch()
