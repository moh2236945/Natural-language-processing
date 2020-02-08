# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary for generating predictions over a set of videos."""

import os
import time

import numpy
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import eval_util
import losses
import readers
import utils

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to load the model files from.")
  
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Model flags.
  
  flags.DEFINE_integer(
      "batch_size", 50,
      "How many examples to process per batch.")
  
  flags.DEFINE_integer("width", 12,
                       "image width")
  flags.DEFINE_integer("Bwidth", 90,
                       "image width")
  flags.DEFINE_integer("height", 36,
                       "image height")
  flags.DEFINE_integer("slices", 15,
                       "slices number")
  
  flags.DEFINE_integer("vocabulary_size", 29,
                       "character's number")
    
  flags.DEFINE_integer("beam_size", 1,
                       "guess number")

  flags.DEFINE_integer("stride", -1,
                       "guess number")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  flags.DEFINE_bool(
      "slice_features", True,
      "If set, then the input should have 4 dimentions ")
  flags.DEFINE_integer("input_chanels", 18,
                       "image width")
  # Other flags.
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string(
      "vocab_path", "vocabulary.txt",
      "Which vocabulary to use in order to help the prediction "
      "in models.py.")




def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    print(files)
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    imageInput, seq_len , target = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch = True,
                            enqueue_many=True))
    return imageInput, seq_len , target

def inference(reader, train_dir, data_pattern, batch_size):
  with tf.Session() as sess:
    #imageInput1, seq_len1 , target1 = get_input_data_tensors(reader, data_pattern, batch_size)
    imageInput, seq_len , target = get_input_data_tensors(reader, data_pattern, batch_size)
    target_dense = tf.sparse_tensor_to_dense(target)
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % train_dir)
    else:
      meta_graph_location = latest_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)
    
    input_tensor = tf.get_collection("input_batch")[0]
    seq_len_tensor = tf.get_collection("seq_len")[0]
    labels_tensor = tf.get_collection("labels")[0]
    predictions_tensor = tf.get_collection("predictions")[0]
    train_batch_tensor = tf.get_collection("train_batch")[0]
    decodedPrediction = []
    for i in range(FLAGS.beam_size):
            decodedPrediction.append(tf.get_collection("decodedPrediction{}".format(i))[0])
    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list   

    sess.run(set_up_init_ops(tf.get_collection_ref(tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()
    
    vocabulary = eval_util.read_vocab(FLAGS.vocab_path)
    vocabulary = sorted(vocabulary, key=lambda word: len(word))

    try:
      while not coord.should_stop() :
          imageInput_val, seq_len_val, target_dense_val = sess.run([imageInput, seq_len , target_dense ])
          
          predictions_val = sess.run(decodedPrediction, 
                                     feed_dict={input_tensor: imageInput_val,seq_len_tensor:seq_len_val,
                                               train_batch_tensor: False})
          #print(predictions_val[0])
          #print(target_dense_val)
          lme, newGuess = eval_util.calculate_models_error_withLanguageModel(predictions_val, 
                                                                                   target_dense_val,
                                                                                   vocabulary, 
                                                                                   FLAGS.beam_size)
          eval_util.show_prediction(predictions_val,target_dense_val,None, 30)
          now = time.time()
          num_examples_processed += len(imageInput_val)
          num_classes = predictions_val[0].shape[1]
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
          
          break

    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.slice_features:
    reader = readers.AIMAggregatedFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes,
               height=FLAGS.height,
               width=FLAGS.width if FLAGS.stride == -1 else FLAGS.Bwidth,
               slices=FLAGS.slices,
                num_classes = FLAGS.vocabulary_size,
                stride = FLAGS.stride,
                input_chanels=FLAGS.input_chanels)
  else:
    raise NotImplementedError()

  

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  inference(reader, FLAGS.train_dir, FLAGS.input_data_pattern,
     FLAGS.batch_size)


if __name__ == "__main__":
  app.run()
