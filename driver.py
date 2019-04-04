import os.path
import tensorflow as tf
import driver_helper
import driver_trainer
import warnings
from distutils.version import LooseVersion
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

import time
import argparse
from moviepy.editor import VideoFileClip
import scipy.misc
import numpy as np
from PIL import Image


def run():

        #Getting arguements
        arg=int(input("\nInput 0 for Training\nInput 1 for Testing\n"))

        #Basic parameters
        print("\nSetting basic parameters")
        num_classes=2
        image_shape=(160,576)
        data_dir='data'
        runs_dir='runs'
        model_path='vgg16/'
        model_name='vgg16model'

        #Tensorflow placeholders
        print("\nSetting tensorflow placeholders")
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')

        #Hyperparameters
        print("\nSetting hyperparameters")
        epochs=25
        learning_rate=0.00009 #tf.placeholder(tf.float32, name='learning_rate')
        batch_size=4

        # #Paths to data
        # print("Setting path to vgg model and training data")
        # vgg_path=os.path.join(data_dir, 'vgg')
        # data_folder=os.path.join(data_dir,'data_dir/training')

        # #generator function to get batches for training
        # print("Getting generator function to get batches of images")
        # get_batches_fn=driver_helper.gen_batch_function(data_folder, image_shape)


        with tf.Session() as sess:
                
                if arg==0:
                        #Path to vgg model
                        vgg_path=os.path.join(data_dir, 'vgg')
                        #Create function to get batches
                        get_batches_fn=driver_helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

                        #Build network using load_vgg, layers, and optimise function
                        print("\nLoading vgg model and customising layers")
                        # input_image, keep_prob, layer3_out, layer4_out, layer7_out = driver_trainer.load_vgg(sess, vgg_path)
                        # fcn8s_out = driver_trainer.layers(layer3_out, layer4_out, layer7_out, num_classes)
                        # logits, train_op, cross_entropy_loss = driver_trainer.optimize(fcn8s_out, correct_label, learning_rate, num_classes)
                        img_input, keep_prob, vgg3, vgg4, vgg7 = driver_trainer.load_vgg(sess, vgg_path)
                        fcn8s_out = driver_trainer.layers(vgg3, vgg4, vgg7, num_classes)
                        logits, train_op, loss = driver_trainer.optimize(fcn8s_out, correct_label, learning_rate,num_classes)

                        #Train NN
                        #sess.run(tf.global_variables_initializer())
                        driver_trainer.train_nn(sess, epochs, batch_size, get_batches_fn, train_op, loss, img_input, correct_label, keep_prob, learning_rate)

                        #Save model result
                        save_input=int(input("\nDo you want to save the model?(0:yes,1:no)\n"))
                        if save_input==0:
                                print("\nSaving trained model in vgg16 folder")
                                saver=tf.train.Saver()
                                save_path=saver.save(sess, model_path+model_name)
                                print("\nModel saved.")
                        else:
                                print("\nSkipped saving model")
                
                elif arg==1:

                        # Load saved model
                        print("\nLoading saved model for testing")
                        saver = tf.train.import_meta_graph(model_path+model_name+'.meta')
                        saver.restore(sess, tf.train.latest_checkpoint(model_path))
                        graph = tf.get_default_graph()
                        img_input = graph.get_tensor_by_name('image_input:0')
                        keep_prob = graph.get_tensor_by_name('keep_prob:0')
                        fcn8s_out = graph.get_tensor_by_name('fcn8s_out:0')
                        logits = tf.reshape(fcn8s_out, (-1, num_classes))

                        # Process test images
                        print("\nProcessing testing images and saving the segmented images")
                        driver_helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                                        logits, keep_prob, img_input)


if __name__ == '__main__':
    run()
