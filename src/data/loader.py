import os
import struct
import numpy as np
import tensorflow as tf

def load_mnist_raw(data_dir: str, kind: str = 'train') -> tuple:
    """Lê os ficheiros binários originais do MNIST."""
    labels_path = os.path.join(data_dir, f'{kind}-labels-idx1-ubyte')
    images_path = os.path.join(data_dir, f'{kind}-images-idx3-ubyte')

    # Ler Labels
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    # Ler Imagens
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 28, 28, 1)

    return images, labels

def create_dataset(data_dir: str, batch_size: int = 256):
    """Cria o pipeline otimizado com base nos raw bytes."""
    X_train, y_train = load_mnist_raw(data_dir, kind='train')
    
    # Normalização
    X_train = X_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    return dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)