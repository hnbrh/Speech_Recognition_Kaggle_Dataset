import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import spectrogram


def recognize(data):

    g = tf.Graph()
    with g.as_default():
        saver = tf.train.import_meta_graph(
            'model_200.ckpt.meta')
    sess = tf.Session(graph=g)
    with sess.as_default():
        saver.restore(sess, 'model_200.ckpt')

    f, t, Sxx = spectrogram(data, window='hamming')
    sigma = np.std(Sxx)
    X_data = Sxx / sigma
    X_image = X_data[:70, :70]
    X_image = np.expand_dims(X_image, axis=0)
    X_input = np.reshape(X_image, (-1, 4900))
    label_names = ['no', 'yes', 'on', 'off', 'down', 'up', 'left', 'right', 'go', 'wow']
    with sess.as_default():
        feed = {'tf_x:0' : X_input, 'is_train:0' : False}
        y_pred = sess.run(['predicted_labels:0'], feed_dict=feed)

    y_pred = np.array(y_pred)
    idx = y_pred[0][0]
    return label_names[idx]

if __name__ == "__main__":
    print(recognize())