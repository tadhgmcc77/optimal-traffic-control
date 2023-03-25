import os
import numpy as np

from tensorflow import keras
from keras import layers
from keras import losses
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.models import load_model


class TrainNeuralNet:
    def __init__(self, num_layers, input_dimensions, output_dimensions, width, batch_size, learning_rate):
        self.input_dimemsions = input_dimensions
        self._output_dimensions = output_dimensions
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._neural_net = self.build_neural_net(num_layers, width)

    # build the neural net to predict Q values from inputs (state)
    def build_neural_net(self, num_layers, width):
        inputs = keras.Input(shape=(self.input_dimemsions,))
        # Create input layer where no.inputs == number of possible states
        x = layers.Dense(width, activation='relu')(inputs)
        # Create the input amount of hidden layers
        for i in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        # No.outputs is = number of possible actions to take | activation is linear because we are values (regression)
        outputs = layers.Dense(self._output_dimensions, activation='linear')(x)

        neural_net = keras.Model(inputs=inputs, outputs=outputs, name='model')
        neural_net.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self._learning_rate))
        return neural_net
    
    def predict_single(self, state):
        state = np.reshape(state, [1, self.input_dimemsions])
        return self._neural_net.predict(state)
    
    def predict_batch(self, states):
        return self._neural_net.predict(states)
    
    def train_batch(self, states, updated_Q):
        self._neural_net.fit(states, updated_Q, epochs=1)

    def save_nerual_net(self, filepath):
        self._model.save(os.path.join(filepath, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(filepath, 'model_structure.png'), show_shapes=True, show_layer_names=True)

        