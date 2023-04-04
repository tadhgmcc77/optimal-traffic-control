import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import sys
from tensorflow import keras
from keras import layers
from keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainNeuralNet:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dimensions, output_dimensions):
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
        self._neural_net.fit(states, updated_Q, epochs=1, verbose=0)

    def save_neural_net(self, filepath):
        self._neural_net.save(os.path.join(filepath, 'trained_model.h5'))
        #plot_model(self._model, to_file=os.path.join(filepath, 'model_structure.png'), show_shapes=True, show_layer_names=True)

    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size
    

class TestModel:
    def __init__(self, input_dimensions, model_path):
        self._input_dim = input_dimensions
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim