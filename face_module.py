import os

import tensorflow as tf
import numpy as np

class FaceModule(object):

    def __init__(self, keras_file):
        self.face_net = tf.keras.models.load_model(keras_file)
        self.face_net.summary()

    def __call__(self, img):
        img = tf.cast(img, dtype=tf.float32) / 255.
        em = self.face_net(img, training=False)
        em = tf.math.l2_normalize(em, axis=1)
        return em.numpy()

    def get_train_data_from_dir(self, reference_dir):
        xs = []
        ys = []
        embeddings = []
        labels = []
        for label_dir in os.listdir(reference_dir):
            label = int(label_dir)
            label_dir = os.path.join(reference_dir, label_dir)
            if not os.path.isdir(label_dir):
                continue
            for img_path in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_path)
                with open(img_path, 'rb') as jpeg_file:
                    jpeg_bytes = jpeg_file.read()
                img = tf.image.decode_jpeg(jpeg_bytes)
                img = tf.image.resize(img, (112, 112))
                img = tf.cast(img, dtype=tf.float32)
                xs.append(img.numpy())
                ys.append(np.array(label))

        xs = np.stack(xs)
        ys = np.stack(ys)
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.batch(32)
        for x, y in ds:
            x /= 255.
            em = self.face_net(x, training=False)
            em = tf.math.l2_normalize(em, axis=1)
            em = em.numpy()
            for a, b in zip(em, y):
                embeddings.append(a.tolist())
                labels.append(b.numpy())
        return embeddings, labels
