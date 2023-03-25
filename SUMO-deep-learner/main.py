from __future__ import absolute_import
from __future__ import print_function

import os
import datetime
from shutil import copyfile

from training import Simulation
from generator import TrafficGenerator
from replay_memory import Replay_memory
from neural_net import TrainNeuralNet
from utils import import_train_configuration, set_sumo, set_train_path
#from visualization import Visualization

if __name__ == "__main__":
    config = import_train_configuration(config_file='config.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    Model = TrainNeuralNet(
        config['num_layers'],
        config['width_layers'],
        config['batch_size'],
        config['learning_rate'],
        input_dimensions=config['num_states'],
        output_dimensions=config['num_actions'],
    )

    Replay_memory = Replay_memory(
        config['memory_size_max'],
        config['memory_size_min'],
    )
    
    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )

    Simulation = Simulation(
        Model,
        Replay_memory,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    while episode < config['total_episodes']:
        print('\n----- Episode', str(episode+1), 'of', str(config['total_episodes']))
        epsilon = 1.0 - (episode / config['total_episodes'])  # set the epsilon for this episode according to epsilon-greedy policy
        simulation_time, training_time = Simulation.run(episode, epsilon)  # run the simulation
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:', round(simulation_time+training_time, 1), 's')
        episode += 1

        print("\n----- Start time:", timestamp_start)
        print("----- End time:", datetime.datetime.now())
        print("----- Session info saved at:", path)
    
        Model.save_model(path)

        copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))