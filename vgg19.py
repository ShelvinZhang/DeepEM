import os,time
import tensorflow as tf
import numpy as np

class Vgg19:
    def __init__(self,args):
        self.args = args
        self.variables_deepem()
        self.build_model()

    def variables_deepem(self):
        self.data_dict = {
            # filter = [kernelsize, kernelsize, last_layser_feature_map_num, feature_map_num]
            # biases = [feature_map_num]
            'conv1_1': {'filter': [3,3,1,  64],  'biases': [64]},
            'conv1_2': {'filter': [3,3,64, 64],  'biases': [64]},
            'conv2_1': {'filter': [3,3,64, 128], 'biases':[128]},
            'conv2_2': {'filter': [3,3,128,128], 'biases':[128]},
            'conv3_1': {'filter': [3,3,128,256], 'biases':[256]},
            'conv3_2': {'filter': [3,3,256,256], 'biases':[256]},
            'conv3_3': {'filter': [3,3,256,256], 'biases':[256]},
            'conv3_4': {'filter': [3,3,256,256], 'biases':[256]},
            'conv4_1': {'filter': [3,3,256,512], 'biases':[512]},
            'conv4_2': {'filter': [3,3,512,512], 'biases':[512]},
            'conv4_3': {'filter': [3,3,512,512], 'biases':[512]},
            'conv4_4': {'filter': [3,3,512,512], 'biases':[512]},
            'conv5_1': {'filter': [3,3,512,512], 'biases':[512]},
            'conv5_2': {'filter': [3,3,512,512], 'biases':[512]},
            'conv5_3': {'filter': [3,3,512,512], 'biases':[512]},
            'conv5_4': {'filter': [3,3,512,512], 'biases':[512]},
            'fc6': {'filter': [], 'biases':[4096]},    # 4096,
            'fc7': {'filter': [4096, 4096], 'biases':[4096]},    # 4096,
            'fc8': {'filter': [4096,    1], 'biases':[1]}   # 1
        }
        

    def build_model(self):
        self.X = tf.placeholder(tf.float32, shape = [None, self.args.boxsize, self.args.boxsize, 1])
        self.Y = tf.placeholder(tf.float32, shape = [None,1])
        self.global_step = tf.Variable(0, name='global_step',trainable=False)

        self.conv1_1 = self.conv_layer(self.X,       "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2,       'pool1')
        print("shape of conv1_1: ", self.conv1_1.shape)
        print("shape of conv1_2: ", self.conv1_2.shape)
        print("shape of pool1:   ", self.pool1.shape)

        self.conv2_1 = self.conv_layer(self.pool1,   "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2,       'pool2')
        print("shape of conv2_1: ", self.conv2_1.shape)
        print("shape of conv2_2: ", self.conv2_2.shape)
        print("shape of pool2:   ", self.pool2.shape)

        self.conv3_1 = self.conv_layer(self.pool2,   "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4,       'pool3')
        print("shape of conv3_1: ", self.conv3_1.shape)
        print("shape of conv3_2: ", self.conv3_2.shape)
        print("shape of conv3_3: ", self.conv3_3.shape)
        print("shape of conv3_4: ", self.conv3_4.shape)
        print("shape of pool3:   ", self.pool3.shape)

        self.conv4_1 = self.conv_layer(self.pool3,   "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4,       'pool4')
        print("shape of conv4_1: ", self.conv4_1.shape)
        print("shape of conv4_2: ", self.conv4_2.shape)
        print("shape of conv4_3: ", self.conv4_3.shape)
        print("shape of conv4_4: ", self.conv4_4.shape)
        print("shape of pool4:   ", self.pool4.shape)

        self.conv5_1 = self.conv_layer(self.pool4,   "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4,       'pool5')
        print("shape of conv5_1: ", self.conv5_1.shape)
        print("shape of conv5_2: ", self.conv5_2.shape)
        print("shape of conv5_3: ", self.conv5_3.shape)
        print("shape of conv5_4: ", self.conv5_4.shape)
        print("shape of pool5:   ", self.pool5.shape)


        self.fc6 = self.fc_layer(self.pool5, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)

        if self.args.is_training and self.args.dropout:
            self.relu6 =  tf.nn.dropout(self.relu6, self.args.dropout_rate)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        if self.args.is_training and self.args.dropout:
            self.relu7 =  tf.nn.dropout(self.relu7, self.args.dropout_rate)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        print("shape of fc6:     ", self.fc6.shape)
        print("shape of fc7:     ", self.fc7.shape)
        print("shape of fc8:     ", self.fc8.shape)
    
        # calculate the accuracy in the training set
        self.logits = self.fc8
        self.pred = tf.nn.sigmoid(self.logits, name="prob")

        if not self.args.is_training:
            return
        if not self.args.regularization:
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.pred, labels = self.Y))
        else:
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels = self.Y)) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc6')) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc7')) + \
                    tf.contrib.layers.l2_regularizer(self.args.reg_rate)(self.get_fc_weight('fc8'))
                  
        self.lr = tf.maximum(1e-12,tf.train.exponential_decay(self.args.learning_rate, self.global_step, self.args.decay_step, self.args.decay_rate, staircase=True))
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.global_step)

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            if name == "fc6":
                self.data_dict['fc6']['filter'] = [dim, 4096]
            
            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.Variable(tf.contrib.layers.xavier_initializer()(self.data_dict[name]['filter']))

    def get_bias(self, name):
        return tf.Variable(tf.zeros(self.data_dict[name]['biases']), name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(tf.contrib.layers.xavier_initializer()(self.data_dict[name]['filter']))
