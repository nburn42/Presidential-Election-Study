import tensorflow as tf
import random
import numpy as np

header = None
linear_header = []
labels_header = []

linear_data = []
data = []
raw_labels = []
labels = []

last_labels = []

mins = []
maxs = []

linear_mins = []
linear_maxs = []


linear_columns = [-1, 10, 7, 6, 4 ,0]

with open("smalldataset.csv", "r") as f:
    for line in f:
        if header is None:
            header = line.strip().split(",")
            labels_header = header[-12:]
            header = header[:-12]
            header.pop(1)

            for i in linear_columns:
                linear_header.append(header.pop(i))

            mins = [99**99 for x in range(len(header))]
            maxs = [0 for x in range(len(header))]
            linear_mins = [99 ** 99 for x in range(len(header))]
            linear_maxs = [0 for x in range(len(header))]
            continue
        d = list(map(int, line.strip().split(",")))
        dd = d[:-12]

        dd.pop(1)

        ld = []
        for i in linear_columns:
            ld.append(dd.pop(i))

        data.append(dd)
        linear_mins = [float(min(x, y)) for x, y in zip(linear_mins, ld)]
        linear_maxs = [float(max(x, y)) for x, y in zip(linear_maxs, ld)]
        linear_data.append(ld)
        mins = [min(x,y) for x,y in zip(mins, dd)]
        maxs = [max(x, y) for x, y in zip(maxs, dd)]
        raw_labels.append(d[-12:])

new_linear_data = []
for line in linear_data:
    dd = []
    for i, l in enumerate(line):
        dd.append((l - linear_mins[i])/(0.0002 + linear_maxs[i] -linear_mins[i]))
    new_linear_data.append(dd)

linear_data = new_linear_data

print(list(zip(header, data[0])))
print(list(zip(header, mins)))
print(list(zip(header, maxs)))
print(list(zip(linear_header, linear_data[0])))
print(list(zip(linear_header, linear_mins)))
print(list(zip(linear_header, linear_maxs)))
print(list(zip(labels_header, raw_labels[-50])))



for ind, d in enumerate(data):
    for i in range(len(d)):
        dd = [0 for x in range(1 + maxs[i] - mins[i])]
        dd[d[i] - mins[i]] = 1
        linear_data[ind].extend(dd)


useful_indexes = [[] for x in labels_header]

for ind, l in enumerate(raw_labels):
    labels.append([])
    last_labels.append([])
    for i in range(len(l)):
        dd = [0 for x in range(2)]
        if(l[i] == 1 or l[i] == 2):
            dd[l[i] - 1] = 1
            useful_indexes[i].append(ind)
        labels[ind].extend(dd)
        if i == len(l) - 1:
            last_labels[ind].extend(dd)


print(map(len, useful_indexes))

final_data = []
final_labels = []

for ind in useful_indexes[-1]:
    final_data.append(linear_data[ind])
    final_labels.append(last_labels[ind])

dataset = list(zip(final_data, final_labels))
random.shuffle(dataset)
test_length = int(len(dataset) * 0.67)

print("test_length", test_length)
train_dataset = dataset[:test_length]
test_dataset = dataset[test_length:]


#### Model ####
input_size = len(train_dataset[0][0])
input_placeholder = tf.placeholder(tf.float32, (None, input_size))

momentum = 0.95
initializer=tf.contrib.layers.xavier_initializer()

layer = input_placeholder
#layer = tf.nn.l2_normalize(input_placeholder, -1)
norm_input = layer

layer_sizes = [100, 100, 100, 100, 2]
last_size = input_size

for i, size in enumerate(layer_sizes):
    weights = tf.get_variable("weight{}".format(i), shape=[last_size, size], initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable("bias{}".format(i), shape=[size], initializer=tf.constant_initializer(value=0.0))
    layer = tf.nn.relu(tf.matmul(layer, weights) + bias)
    #layer = tf.layers.batch_normalization(layer, momentum=momentum)
    last_size = size


# layer = tf.nn.relu(tf.layers.dense(layer, 1000, kernel_initializer=initializer))
# #layer = tf.layers.batch_normalization(layer, momentum=momentum)
# #layer = tf.nn.dropout(layer, 0.5)
# layer = tf.nn.relu(tf.layers.dense(layer, 1000, kernel_initializer=initializer))
# #layer = tf.layers.batch_normalization(layer, momentum=momentum)
# #layer = tf.nn.dropout(layer, 0.5)
# layer = tf.nn.relu(tf.layers.dense(layer, 1000, kernel_initializer=initializer))
# #layer = tf.layers.batch_normalization(layer, momentum=momentum)
# #layer = tf.nn.dropout(layer, 0.5)
# layer = tf.nn.relu(tf.layers.dense(layer, 1000, kernel_initializer=initializer))
# #layer = tf.layers.batch_normalization(layer, momentum=momentum)
# #layer = tf.nn.dropout(layer, 0.5)

logits = tf.nn.softmax(layer)

label_placeholder = tf.placeholder(tf.float32, (None, 2))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_placeholder, logits=logits))

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
train = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in range(5000):
        batch = random.sample(train_dataset, 2000)
        inputs_batch, labels_batch = zip(*batch)
        loss_output, prediction_output, _ , norm = sess.run([loss, logits, train, norm_input], feed_dict={input_placeholder: inputs_batch, label_placeholder: labels_batch})

        print("input", zip(inputs_batch[0], norm[0]))
        debug_output = zip(prediction_output[:10].tolist(), labels_batch[:10])
        for d_o in debug_output:
            print(d_o)


        accuracy = np.mean(labels_batch == prediction_output)

        print("step", step, "train", "loss", loss_output, "accuracy", accuracy)