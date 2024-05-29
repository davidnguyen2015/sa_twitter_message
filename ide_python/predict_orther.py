import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import os
import sys
import csv
import datetime
import data_helpers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def evaluation(folder_data, folder_checkpoint):
    # Parameters
    # ==================================================
    # Data Parameters
    tf.flags.DEFINE_string("positive_data_file_test", os.path.join(folder_data, "rt-polarity.pos"), "Data source for the positive data.")
    tf.flags.DEFINE_string("neutral_data_file_test", os.path.join(folder_data, "rt-polarity.neu"), "Data source for the neutral data.")
    tf.flags.DEFINE_string("negative_data_file_test", os.path.join(folder_data, "rt-polarity.neg"), "Data source for the positive data.")

    # Eval Parameters
    tf.flags.DEFINE_integer("batch_size_test", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_string("checkpoint_dir_test", "./runs/" + folder_checkpoint + "/checkpoints/", "Checkpoint directory from training run")
    tf.flags.DEFINE_boolean("eval_train_test", True, "Evaluate on all training data")
    tf.flags.DEFINE_integer("embedding_dim_test", 300, "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_integer("max_pool_size_test", 4, "Number of max_pool_size to maxpool (default: 4)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement_test", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement_test", False, "Log placement of ops on devices")


    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    # print("\nParameters:")
    # for attr, value in sorted(FLAGS.__flags.items()):
    #     print("{}={}".format(attr.upper(), value))
    # print("")

    # CHANGE THIS: Load data. Load your own data here
    if FLAGS.eval_train_test:
        x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file_test, FLAGS.neutral_data_file_test, FLAGS.negative_data_file_test)
        y_test = np.argmax(y_test, axis=1)
    else:
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]

    # Map data into vocabulary
    vocab_path = os.path.join(FLAGS.checkpoint_dir_test, "..", "vocab")
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nPredicting...\n")

    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir_test)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement_test,
          log_device_placement=FLAGS.log_device_placement_test)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob_cnn = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            batch_size = graph.get_operation_by_name("batch_size").outputs[0]
            pad = graph.get_operation_by_name("pad").outputs[0]
            real_len = graph.get_operation_by_name("real_len").outputs[0]

            def real_len_test(batches):
                return [np.ceil(np.argmin(batch + [0]) * 1.0 / FLAGS.max_pool_size_test) for batch in batches]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size_test, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch,
                                                           dropout_keep_prob_cnn: 1.0,
                                                           batch_size: len(x_test_batch),
                                                           pad: np.zeros([len(x_test_batch), 1, FLAGS.embedding_dim_test , 1]),
                                                           real_len: real_len_test(x_test_batch)})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    if y_test is not None:
        correct_predictions = float(sum(all_predictions == y_test))
        #print("Total number of test examples: {}".format(len(y_test)))
        #print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
        number_test = len(y_test)
        accuracy = correct_predictions/float(number_test)
        print("Total number of test examples: {}".format(number_test))
        print("Accuracy: {:g}".format(accuracy))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    out_path = os.path.join(FLAGS.checkpoint_dir_test, "..", "prediction.csv")
    print("Saving evaluation to {0}".format('./runs/' + folder_checkpoint + '/'))
    print(predictions_human_readable)
    with open(out_path, 'w', encoding='utf-8') as f:
        csv.writer(f).writerows(predictions_human_readable)

    return number_test, accuracy

def run_predict(folder_data, file_log, folder_checkpoint):
    start_time = datetime.datetime.now()
    print("Start predicting statement analysis twitter messages time: ", start_time.strftime('%Y-%m-%d %H:%M:%S'))

    number_test, accuracy = evaluation(folder_data, folder_checkpoint)

    print('Prediction for data in folder ' + folder_data + ' successfully.')

    end_time = datetime.datetime.now()
    time_run = end_time - start_time
    result = divmod(time_run.days * 86400 + time_run.seconds, 60)
    print("End predicting statement analysis twitter messages time: ", end_time.strftime('%Y-%m-%d %H:%M:%S'))
    print("Total time predict finish (minutes, seconds): ", result)

    # Save result in file log.
    log = open("./log/" + file_log + ".txt", "a")
    log.write("----------------------------" + "\n")
    log.write("Test model with data:  " + str(folder_data) + "\n")
    log.write("Total time test finish (minutes, seconds): " + str(result) + "\n")
    log.write("Total number of test examples: " + str(number_test) + "\n")
    log.write("Accuracy: " + str(accuracy) + "\n")
    log.write("\n")
    log.close()


if __name__ == "__main__":

    run_predict(sys.argv[1], sys.argv[2], sys.argv[3])
    print("----------------------------")
