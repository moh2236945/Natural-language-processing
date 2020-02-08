# -*- coding: UTF-8 -*-
# python3.5(tensorflow)：C:\Users\Dr.Du\AppData\Local\conda\conda\envs\tensorflow\python.exe
# python3.6：C:\ProgramData\Anaconda3
# -*- coding: utf-8 -*-
# @Time    : 2018/8/30 23:31
# @Author  : tuhailong
import tensorflow as tf
import nmupy as np

class DPCNN(object):

	def region_embed(self, input):
		input_list = []
		for reigon_sieze in self.region_size_list:
			pass
		return tf.concat(3, input_list)

	def change_dim_conv(self, input):
		pass

	def conv(self, input):
		pass

    def block_conv(self, input):
        pass

    def fc(self, input):
    	pass

    def __init__(self, BASIC_INFO):
        self.feature_size = BASIC_INFO["feature_size"]
        self.vocab_size = BASIC_INFO["vocab_size"]
        self.embedding_size = BASIC_INFO["embedding_size"]
        self.region_size_list = BASIC_INFO["region_size_list"]  #[1,2,3,4]
        self.lambda_l2 = ASIC_INFO["lambda_l2"]

        self.input_x = tf.placeholder(tf.int32, [None,sentence_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None,num_classes], name="input_y")
  
        #l2 regularization
        l2_loss = tf.constant(0.0)

        #embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("embedding"):
			self.embedding_W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size],-1.0,1.0),name="embedding_W")   
			self.embedding_input_x = tf.nn.embedding_lookup(self.embedding_W, self.input_x)     #?*len*embed

			print("embedding_input_x:", self.embedding_input_x.get_shape()) 

		with tf.variable_scope("pre_block"):

			input_x = region_embed(input_x)	                       #?*len*embed*4(region_channel)
			#conv pre_act in shortcut 
			input_x = change_dim_conv(self.embedding_input_x)	    #?*len*embed*250
			input_px = input_x 
			#conv
  			input_x = self.conv(input_x)           #?*len*embed*250
  			#act_fun
  			input_x = tf.nn.relu(input_x)
  			#conv
  			input_x = self.conv(input_x)           #?*len*embed*250
  			#act_fun
  			input_x = tf.nn.relu(input_x)
  			#shortcut 
  			input_x = input_x + input_px

  		with tf.variable_scope("conv_block"):
  			for i in range(num_block):
                if input_x.get_shape()[2] < 2:
                	input_x = self.block_conv(input_x)
                else:
                	break

        with tf.variable_scope("fc"):
            flatten_size = np.prod(input_x)
            input_x = tf.reshape(input_x, [-1, flatten_size])
            w = tf.get_variable('w', [flatten_size, num_classes], initializer=    )
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(1.0))
            self.output_x = tf.matmul(input_x, w) + b
           
        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
          	self.predictions = tf.argmax(self.output_x, 1, name="predictions")
           	losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_x, labels=self.input_y)
           	l2_loss = 
           	self_loss = tf.reduce_mean(loss) + self.lambda_l2*l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





















