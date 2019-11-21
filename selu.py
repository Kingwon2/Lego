import tensorflow as tf
import random
import load_data
import matplotlib.pyplot as plt

# label number to label name
def convert_label_name(label):
    label_names = ["2357 Brick corner 1x2x2", "3003 Brick 2x2", "3004 Brick 1x2", "3005 Brick 1x1", "3022 Plate 2x2", "3023 Plate 1x2",
                   "3024 Plate 1x1", "3040 Roof Tile 1x2x45deg", "3069 Flat Tile 1x2", "3673 Peg 2M", "3713 Bush for Cross Axle", "3794 Plate 1X2 with 1 Knob",
                   "6632 Technic Lever 3M", "11214 Bush 3M friction with Cross axle", "18651 Cross Axle 2M with Snap friction", "32123 half Bush"]
    return label_names[int(label)]

tf.set_random_seed(779)  # for reproducibility

# load data
try:
    train_file_path = './train_data.csv'
    test_file_path = './test_data.csv'
    train_datas = load_data.load_and_preprocess_data(train_file_path)
    test_datas = load_data.load_and_preprocess_data(test_file_path)
except:
    print('우선 preprocess.py를 실행하셔야 합니다.')
    exit(1)

print('train Input X Shape:' + str(train_datas.data[...,-16:].data.shape))
print('train Label Y Shape:' + str(train_datas.data[...,:-16].data.shape))
def selu(x):
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def dropout_selu(x, keep_prob, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = tf.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * tf.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
                                lambda: dropout_selu_impl(x, keep_prob, alpha, noise_shape, seed, name),
                                lambda: array_ops.identity(x))
#length of data
len_training_data = len(train_datas.data)
len_test_data =len(test_datas.data)
nb_classes = 16

# parameters
training_epochs = 50
batch_size = 512
learning_rate = 0.001

# input place holders
X = tf.placeholder(tf.float32, [None, 2500])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# weights & bias for nn layers
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[2500,1250],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([1250]))
L1 = selu(tf.matmul(X, W1) + b1)
L1 = dropout_selu(L1, keep_prob=keep_prob)
W2 = tf.get_variable("W2", shape=[1250,800],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([800]))
L2 = selu(tf.matmul(L1, W2) + b2)
L2 = dropout_selu(L2, keep_prob=keep_prob)
W3 = tf.get_variable("W3", shape=[800,800],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([800]))
L3 = selu(tf.matmul(L2, W3) + b3)
L3 = dropout_selu(L3, keep_prob=keep_prob)
W4 = tf.get_variable("W4", shape=[800,800],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([800]))
L4 = selu(tf.matmul(L3, W4) + b4)
L4 = dropout_selu(L4, keep_prob=keep_prob)
W5 = tf.get_variable("W5", shape=[800,800],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([800]))
L5 = selu(tf.matmul(L4, W5) + b5)
L5 = dropout_selu(L4, keep_prob=keep_prob)
W6 = tf.get_variable("W6", shape=[800,nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([nb_classes]))

logits = tf.matmul(L5, W6) + b6
hypothesis = tf.nn.softmax(logits)

# Cost function & Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y)) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len_training_data / batch_size)

        for i in range(total_batch):
            # 원본 학습 데이터로부터 batch size만큼 추출해가며 학습
            batch = train_datas.next_batch(batch_size=batch_size)
            batch_x = batch[...,:-16]
            batch_y = batch[...,-16:]

            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    avg_acc=0
    batch_size = 512
    total_batch = int(len_test_data / batch_size)

    for i in range(total_batch):
        # 원본 테스트 데이터로부터 batch size만큼 추출해가며 정확도 계산
        batch = test_datas.next_batch(batch_size=batch_size)
        batch_x = batch[...,:-16]
        batch_y = batch[...,-16:]
        acc = sess.run(accuracy, feed_dict={X:batch_x, Y: batch_y, keep_prob: 1})
        avg_acc += acc / total_batch

    print("Accuracy: ", avg_acc)


    # Get one and predict
    print("\n<무작위로 1개 추출 후 예측>")
    r = random.randint(0, len_test_data - 1)
    one_data = test_datas.data[r:r+1]

    # get label value
    label = sess.run(tf.argmax(one_data[...,-16:], 1)) # [0,0,1...0] -> 3 변환
    label_name = convert_label_name(label)  # 0 -> 2357 Brick corner 1x2x2 변환
    print("Label: ", label, label_name)

    # get predict value
    pred = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: one_data[...,:-16],keep_prob: 1})
    pred_name = convert_label_name(pred)
    print("Prediction: ", pred, pred_name)

    # numpy array로 부터 이미지를 출력하는 부분
    plt.imshow(
        one_data[...,:-16].reshape(50,50),
        cmap='Greys',
        interpolation='nearest')
    plt.show()


