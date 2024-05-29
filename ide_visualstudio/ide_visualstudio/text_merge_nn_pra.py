import tensorflow as tf


class Text_NN(object):
    """
    A neural network mixed (CNN, RNN) for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Uses an embedding layer, GRUCell and output states of RNN
    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 filter_sizes, num_filters, hidden_unit, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob_cnn = tf.placeholder(tf.float32, name="dropout_keep_prob_cnn")
        self.dropout_keep_prob_rnn = tf.placeholder(tf.float32, name="dropout_keep_prob_rnn")
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            emb = tf.expand_dims(self.embedded_chars, -1)

        # Zero paddings so that the convolution output have dimension batch x sequence_length x emb_size x channel
        num_prio = (filter_sizes[0] - 1) // 2
        num_post = (filter_sizes[0] - 1) - num_prio
        pad_prio = tf.concat([self.pad] * num_prio, 1, name="pad_prio")
        pad_post = tf.concat([self.pad] * num_post, 1, name="pad_post")
        self.embedded_chars_pad = tf.concat([pad_prio, emb, pad_post], 1, name="embedded_chars_pad")

        # Convolution neural network (Convolution + maxpool layer for filter size)
        pooled_outputs = []
        with tf.name_scope("conv-maxpool-%s" % filter_sizes[0]):
            # Convolution Layer
            filter_shape = [filter_sizes[0], embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.embedded_chars_pad,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")

            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            k = sequence_length - filter_sizes[0] + 1

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, k, 1, 1],
                strides=[1, k, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

        # Combine all the pooled features
        self.h_pool = tf.concat(pooled_outputs, 2, name="h_pool")
        # Add dropout
        self.h_drop_maxpool = tf.nn.dropout(self.h_pool, self.dropout_keep_prob_cnn, name="max_pooling")

        # Recurrent neural network (Gated Recurrent Unit Cell - GRNN)
        gru_cell = tf.contrib.rnn.GRUCell(num_units=hidden_unit)
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell, output_keep_prob=self.dropout_keep_prob_rnn)

        num_word = self.embedded_chars_pad.get_shape()[1].value
        embedded_rnn = tf.reshape(self.embedded_chars_pad, [-1, num_word, embedding_size], name="embedded_rnn")
        inputs = [tf.squeeze(input_, [1], name="input_rnn") for input_ in tf.split(embedded_rnn, num_word, 1)]
        outputs, state = tf.contrib.rnn.static_rnn(gru_cell, inputs, dtype=tf.float32)

        # Merge result of two neural CNN (max pooling) and RNN (outputs)
        merge_value = self.h_drop_maxpool.get_shape()[1].value * self.h_drop_maxpool.get_shape()[2].value
        rnn_output = tf.stack(outputs, axis=1, name="rnn_output")
        cnn_output = tf.reshape(self.h_drop_maxpool, [-1, merge_value, num_filters], name="cnn_output")

        merge_outputs = tf.concat([cnn_output, rnn_output], 1, name="merge_outputs")
        outputs_final = [tf.squeeze(input_, [1], name="outputs_final") for input_ in
                         tf.split(merge_outputs, merge_outputs.get_shape()[1].value, 1)]

        print("Word embedding:", emb.get_shape)
        print("Word embedding after add pad:", self.embedded_chars_pad.get_shape)
        print("Max-pool:", self.h_drop_maxpool.get_shape)
        print("Output CNN:", cnn_output.get_shape)
        print("Output GRNN:", rnn_output.get_shape)
        print("Output merge:", merge_outputs.get_shape)
        print("Outputs final:", outputs_final)

        # Final (unnormalized) scores and predictions (CNN + GRNN)
        output = outputs_final[0]
        with tf.variable_scope('output_array'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, hidden_unit], tf.float32)
            for i in range(1, len(outputs_final)):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                mat = tf.matmul(ind, one)
                output = tf.add(tf.multiply(output, mat), tf.multiply(outputs_final[i], 1.0 - mat))

        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal([hidden_unit, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
