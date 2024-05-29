import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import os
import sys
import time
import datetime
import data_helpers
from text_merge_nn_pra import Text_NN
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def training_data(folder_data, timestamp, flg_model, dev_sample_percentage,
                  embedding_dim, filter_sizes, num_filters, hidden_unit,
                  dropout_keep_prob_cnn, dropout_keep_prob_rnn, l2_reg_lambda, max_pool_size,
                  batch_size, num_epochs, evaluate_every, checkpoint_every, num_checkpoints):

    tf.reset_default_graph()

    # Parameters
    # ==================================================
    # Data loading params
    tf.flags.DEFINE_float("dev_sample_percentage", dev_sample_percentage, "Percentage of the training data to use for validation")
    tf.flags.DEFINE_string("positive_data_file", os.path.join(folder_data, "rt-polarity.pos"),
                           "Data source for the positive data.")
    tf.flags.DEFINE_string("neutral_data_file", os.path.join(folder_data, "rt-polarity.neu"),
                           "Data source for the neutral data.")
    tf.flags.DEFINE_string("negative_data_file", os.path.join(folder_data, "rt-polarity.neg"),
                           "Data source for the negative data.")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", embedding_dim, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", filter_sizes, "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", num_filters, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob_cnn", dropout_keep_prob_cnn, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("dropout_keep_prob_rnn", dropout_keep_prob_rnn, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", l2_reg_lambda, "L2 regularization lambda (default: 0.0)")
    tf.flags.DEFINE_integer("hidden_unit", hidden_unit, "Number of hidden layer for RNN model (default: 128)")
    tf.flags.DEFINE_integer("max_pool_size", max_pool_size, "Number of max_pool_size to maxpool (default: 4)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", num_epochs, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", evaluate_every,
                            "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", checkpoint_every, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", num_checkpoints, "Number of checkpoints to store (default: 5)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags() 'TODO
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    # Load data
    print("Loading data...")

    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.neutral_data_file,
                                                  FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = None
            print('Training data with model neural is CNN + GRNN, merge maxpool and ouput')

            model = Text_NN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                hidden_unit=FLAGS.hidden_unit,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            # timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def real_len_train(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / FLAGS.max_pool_size) for batch in batches]

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob_cnn: FLAGS.dropout_keep_prob_cnn,
                    model.dropout_keep_prob_rnn: FLAGS.dropout_keep_prob_rnn,
                    model.batch_size: len(x_batch),
                    model.pad: np.zeros([len(x_batch), 1, FLAGS.embedding_dim, 1]),
                    model.real_len: real_len_train(x_batch)
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob_cnn: 1.0,
                    model.dropout_keep_prob_rnn: 1.0,
                    model.batch_size: len(x_batch),
                    model.pad: np.zeros([len(x_batch), 1, FLAGS.embedding_dim, 1]),
                    model.real_len: real_len_train(x_batch)
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            sess.close()

def run_train(folder_data, file_log, arg_value):
    # Model Hyperparameters
    dev_sample_percentage = .3
    embedding_dim = 300
    filter_sizes = "3"
    num_filters = 128
    dropout_keep_prob_cnn = 0.5
    dropout_keep_prob_rnn = 0.5
    l2_reg_lambda = 1.0
    hidden_unit = 128
    max_pool_size = 4

    # Training parameters
    batch_size = 64
    num_epochs = 200
    evaluate_every = 100
    checkpoint_every = 100
    num_checkpoints = 5

    # Set value to predict argument correct
    # l2_reg_lambda = arg_value
    # num_epochs = arg_value
    model_use = arg_value

    timestamp = str(int(time.time()))
    folder_train = folder_data + "train"

    start_time = datetime.datetime.now()
    print('Start training statement analysis twitter messages time: ', start_time.strftime('%Y-%m-%d %H:%M:%S'))

    training_data(folder_train, timestamp, model_use, dev_sample_percentage,
                  embedding_dim, filter_sizes, num_filters, hidden_unit,
                  dropout_keep_prob_cnn, dropout_keep_prob_rnn, l2_reg_lambda, max_pool_size,
                  batch_size, num_epochs, evaluate_every, checkpoint_every, num_checkpoints)

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print('Checkpoint save folder ', out_dir)
    print('Run terminal to view tensorboard: tensorboard --logdir=', out_dir.strip())
    print('and view browser: http://localhost:6006/')
    print('Training for data in folder ' + folder_train + ' successfully.')

    end_time = datetime.datetime.now()
    time_run = end_time - start_time
    result = divmod(time_run.days * 86400 + time_run.seconds, 60)
    print("End training statement analysis twitter messages time: ", end_time.strftime('%Y-%m-%d %H:%M:%S'))
    print("Total time train finish (minutes, seconds): ", result)
    
    print("Optimize with argument:  " + arg_value)

    # Run file predict to test result.
    log = open("./log/" + file_log + ".txt", "a")
    log.write("----------------------------" + "\n")
    log.write("Checkpoint save folder: " + out_dir + "\n")
    log.write("Optimize with argument:  " + str(arg_value) + "\n")
    log.write("Total time train finish (minutes, seconds): " + str(result) + "\n")
    log.write("\n")
    log.close()

    folder_test = folder_data + "test"
    file = "./predict.py"
    subprocess.call([sys.executable, file, folder_test, file_log, str(timestamp)])

    folder_test = folder_data + "train"
    file = "./predict.py"
    subprocess.call([sys.executable, file, folder_test, file_log, str(timestamp)])


if __name__ == "__main__":

    run_train(sys.argv[1], sys.argv[2], sys.argv[3])
    print("----------------------------")

    # log_file = "result_20170523"
    # folder_data = "./data/data_2/"
    # arg_value = 'Mix'
    # run_tran(folder_data, log_file, arg_value)

