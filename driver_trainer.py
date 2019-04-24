import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
import os
import time

LEARN_RATE = 9e-5

def load_vgg(sess, vgg_path):
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
        graph = tf.get_default_graph()
        image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

        return image_input, keep_prob, layer3_out, layer4_out, layer7_out

print("\nPassed load_vgg function")


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

        # Kernel parameters
        kL2Reg = 0.001
        kInitSTD = 0.01

        # Apply 1x1 convolution to VGG layer 7 to reduce # of classes to num_classes
        score_fr = tf.layers.conv2d(vgg_layer7_out, num_classes,
                kernel_size=1, strides=1, padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                kernel_initializer=tf.truncated_normal_initializer(stddev=kInitSTD),
                name='score_fr')

        # Upsample 2x by transposed convolution
        upscore2 = tf.layers.conv2d_transpose(score_fr, num_classes,
                        kernel_size=4, strides=2, padding='same',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                        kernel_initializer=tf.zeros_initializer,
                        name='upscore2')

        # Rescale VGG layer 4 (max pool) for compatibility as a skip layer
        scale_pool4 = tf.multiply(vgg_layer4_out, 0.01, name='pool4_out_scaled')

        # Apply 1x1 convolution to rescaled VGG layer 4 to reduce # of classes
        score_pool4 = tf.layers.conv2d(scale_pool4, num_classes,
                kernel_size=1, strides=1, padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                kernel_initializer=tf.truncated_normal_initializer(stddev=kInitSTD),
                name='score_pool4')

        # Add skip layer from VGG layer 4
        fuse_pool4 = tf.add(upscore2, score_pool4)

        # Upsample 2x by transposed convolution
        upscore_pool4 = tf.layers.conv2d_transpose(fuse_pool4, num_classes,
                        kernel_size=4, strides=2, padding='same',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                        kernel_initializer=tf.zeros_initializer,
                        name='upscore_pool4')

        # Rescale VGG layer 3 (max pool) for compatibility as a skip layer
        scale_pool3 = tf.multiply(vgg_layer3_out, 0.0001, name='pool3_out_scaled')

        # Apply 1x1 convolution to rescaled VGG layer 3 to reduce # of classes
        score_pool3 = tf.layers.conv2d(scale_pool3, num_classes,
                kernel_size=1, strides=1, padding='same',
                kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                kernel_initializer=tf.truncated_normal_initializer(stddev=kInitSTD),
                name='score_pool3')

        # Add skip layer from VGG layer 3
        fuse_pool3 = tf.add(upscore_pool4, score_pool3)

        # Upsample 8x by transposed convolution
        upscore8 = tf.layers.conv2d_transpose(fuse_pool3, num_classes,
                        kernel_size=16, strides=8, padding='same',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(kL2Reg),
                        kernel_initializer=tf.zeros_initializer,
                        name='upscore8')

        # Add identity layer to name output
        fcn8s_out = tf.identity(upscore8, name='fcn8s_out')

        #tf.Print(fcn8_out, [tf.shape(fcn8_out)])

        return fcn8s_out

print("\nPassed layers function")


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
        # Reshape logits and labels
        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        labels = tf.reshape(correct_label, (-1, num_classes))

        # Calculate softmax cross entropy loss and regularization loss operations
        cross_entropies = tf.nn.softmax_cross_entropy_with_logits_v2(
                                                        logits=logits, labels=labels)
        cross_entropy_loss = tf.reduce_mean(cross_entropies)

        l2_reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.reduce_sum(l2_reg_losses)

        total_loss = cross_entropy_loss + regularization_loss

        # Add loss to TensorBoard summary logging
        tf.summary.scalar('loss', total_loss)

        # Set up Adam optimizer and training operation to minimize total loss
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step,
                                name='train_op')

        return logits, train_op, total_loss


print("\nPassed Optimizer")

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,cross_entropy_loss, input_image, correct_label, keep_prob,learning_rate):
        # Set up TensorBoard logging output
        tb_out_dir = os.path.join('tb/', str(time.time()))
        tb_merged = tf.summary.merge_all()
        #train_writer = tf.summary.FileWriter(tb_out_dir, sess.graph) # with graph
        train_writer = tf.summary.FileWriter(tb_out_dir) # without graph

        # Initialize any uninitialized variables
        sess.run(tf.global_variables_initializer())

        # Train network
        for epoch in range(epochs):
                print("Epoch #", epoch+1)

                for image_batch, label_batch in get_batches_fn(batch_size):
                        feed_dict = {input_image: image_batch,
                                        correct_label: label_batch,
                                        keep_prob: 0.5}

                        # Run training step on each batch
                        _, loss_value, summary = sess.run([train_op, cross_entropy_loss,
                                                                tb_merged], feed_dict=feed_dict)

                        # Log loss for each global step
                        step = tf.train.global_step(sess, tf.train.get_global_step())
                        train_writer.add_summary(summary, step)
                        print("  Step", step, "loss =", loss_value)


print("\nPassed training function") 

def conv_hff(vgg_layer, kernel_size, n_split, num_classes):

        # Initialize parameters
        dilation_rate = 2
        branch = list()
        feature_map = None

        # Split the computation into n_split branches
        for i in range(n_split):
                layer_fr = tf.layers.conv2d(vgg_layer, num_classes,kernel_size=1, strides=1, padding='same', dilation_rate=2)
                branch.append(layer_fr)
                dilation_rate += 1
        
        # Concatenate feature maps hierarchically
        feature_map = branch[0]
        for i in range(1,len(branch)-2):
                feature_map += branch[i]

        return feature_map

print("\nPassed HFF function")
