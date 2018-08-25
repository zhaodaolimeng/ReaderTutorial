import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper


class MatchLSTMModel:

    def __int__(self):
        """
        Steps:
        1. content and question embedding
        2. encoding
        3. attention match
        4. boundary PtrNet
        """

        num_units = 200
        num_layers = 3
        dropout = tf.placeholder(tf.float32)

        cells = []
        for _ in range(num_layers):
            cell = GRUCell(num_units)  # Or LSTMCell(num_units)
            cell = DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)


        pass

    def forward(self):

        pass


if __name__ == '__main__':


    pass