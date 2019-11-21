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

#length of data
len_training_data = len(train_datas.data)
len_test_data =len(test_datas.data)
nb_classes = 16

# parameters
training_epochs = 20
batch_size = 412
learning_rate = 0.001

# input place holders
X = tf.placeholder(tf.float32, [None, 2500])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# weights & bias for nn layers


W1 = tf.get_variable("W1", shape=[2500,1250],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([1250]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.get_variable("W2", shape=[1250,800],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([800]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
W3 = tf.get_variable("W3", shape=[800,512],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
W4 = tf.get_variable("W4", shape=[512,256],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([256]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
W5 = tf.get_variable("W5", shape=[256,256],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([256]))
L5 = tf.nn.relu(tf.matmul(L3, W4) + b4)
W6 = tf.get_variable("W6", shape=[256,nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([nb_classes]))
L6 = tf.nn.relu(tf.matmul(L3, W4) + b4)
W7 = tf.get_variable("W7", shape=[256,256],
                     initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([256]))
L7 = tf.nn.relu(tf.matmul(L3, W4) + b4)
W8 = tf.get_variable("W8", shape=[256,nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([nb_classes]))

logits = tf.matmul(L7, W8) + b8
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

            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y})
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
        acc = sess.run(accuracy, feed_dict={X:batch_x, Y: batch_y})
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
    pred = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: one_data[...,:-16]})
    pred_name = convert_label_name(pred)
    print("Prediction: ", pred, pred_name)

    # numpy array로 부터 이미지를 출력하는 부분
    plt.imshow(
        one_data[...,:-16].reshape(50,50),
        cmap='Greys',
        interpolation='nearest')
    plt.show()


