import tensorflow as tf


"""
NO FLAGS
"""


if __name__ == "__main__":

    pred = dynamicRNN(x, seqlen, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()


    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        for step in range(1, training_steps+1):
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           seqlen: batch_seqlen})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                    seqlen: batch_seqlen})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy
        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))
