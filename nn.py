import tensorflow as tf
import random
import numpy as np

from dataset import dataset
from model import model

data = dataset()

input_size = len(data.train_dataset[0][0])
nn = model(input_size)


best_accuracy = 0
best_step = 0
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    for step in range(500):
        batch = random.sample(data.train_dataset, 2000)
        inputs_batch, labels_batch = zip(*batch)
        loss_output, prediction_output, _ = sess.run([nn.loss, nn.logits, nn.train], feed_dict={nn.input_placeholder: inputs_batch, nn.label_placeholder: labels_batch})

        if step % 50 == 0:
            eval = random.sample(data.test_dataset, 1000)
            inputs_eval, labels_eval = zip(*eval)



            loss_eval, prediction_eval = sess.run([nn.loss, nn.logits],
                                                  feed_dict={nn.input_placeholder: inputs_eval,
                                                             nn.label_placeholder: labels_eval})

            debug_output = zip(prediction_output[:10].tolist(), labels_batch[:10])
            for d_o in debug_output:
                print("train\n", d_o)
            debug_eval = zip(prediction_eval[:10].tolist(), labels_eval[:10])
            for d_o in debug_eval:
                print("eval\n", d_o)

            accuracy = np.mean(labels_batch == np.rint(prediction_output))
            accuracy_eval = np.mean(labels_eval == np.rint(prediction_eval))

            print("step", step, "train", "loss", loss_output, "accuracy", accuracy, "eval", "loss", loss_eval, "accuracy", accuracy_eval)
            if (best_accuracy < accuracy_eval):
                best_accuracy = accuracy_eval
                best_step = step

                save_path = saver.save(sess, "models/smallA_{}.ckpt".format(step))
                #print("Model saved in file: %s" % save_path)


    # eval
    saver.restore(sess, "models/smallA_{}.ckpt".format(best_step))

    eval = data.test_dataset
    inputs_eval, labels_eval = zip(*eval)

    loss_eval, prediction_eval = sess.run([nn.loss, nn.logits],
                                          feed_dict={nn.input_placeholder: inputs_eval,
                                                     nn.label_placeholder: labels_eval})

    accuracy_eval = np.mean(labels_eval == np.rint(prediction_eval))

    print("Final Accuracy", best_step, "loss", loss_eval, "accuracy",
          accuracy_eval)
    if (best_accuracy < accuracy_eval):
        best_accuracy = accuracy_eval
        best_step = step

    with open("final_output", "w") as f:
        f.write(str(zip(inputs_eval, labels_eval, prediction_eval)))