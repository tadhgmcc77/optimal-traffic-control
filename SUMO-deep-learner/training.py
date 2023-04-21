import traci
import numpy as np
import random
import timeit
import os
from utils import set_phaseID



config = set_phaseID(config_file='config.ini')

networkID = config['networkID']

if networkID == 0:

    # phase codes based on environment.net.xml
    PHASE_NS_GREEN = 0  # action 0 code 00
    PHASE_NS_YELLOW = 1
    PHASE_NSL_GREEN = 2  # action 1 code 01
    PHASE_NSL_YELLOW = 3
    PHASE_EW_GREEN = 4  # action 2 code 10
    PHASE_EW_YELLOW = 5
    PHASE_EWL_GREEN = 6  # action 3 code 11
    PHASE_EWL_YELLOW = 7

elif networkID == 1:
    # phase codes based on simple-intersection.net.xml
    PHASE_NS_GREEN = 0  # action 0 code 00
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN = 2  # action 1 code 01
    PHASE_EW_YELLOW = 3

elif networkID == 2:
    # phase codes based on simple-roundabout.net.xml
    PHASE_NS_GREEN = 0 # action 0 code 00
    PHASE_NS_YELLOW = 1
    PHASE_NS_RED = 2
    PHASE_NS_THROUGH = 3 # action 1 code 01
    PHASE_NS_THROUGH_YELLOW = 4
    PHASE_NS_THROUGH_RED = 5

elif networkID == 3:
    # phase codes based on kinsale.net.xml
    PHASE_NS_GREEN = 0  # action 0 code 00
    PHASE_NS_YELLOW = 1
    PHASE_EW_GREEN = 2  # action 1 code 01
    PHASE_EW_YELLOW = 3


class Simulation:
    def __init__(self, neural_net, replay_memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._neural_net = neural_net
        self._replay_memory = replay_memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._training_epochs = training_epochs
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []


    def run_simulation(self, episode, epsilon):
        start_time = timeit.default_timer()

        #Generate traffic and route file for this simulation + configure sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        #initialise variables for simulation
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_total_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        print("max = ", self._max_steps)

        while self._step < self._max_steps:
            # get state of intersection
            #print("step--", self._step)
            current_state = self._get_state()

            # calculate reward of previous action (change in total waiting time between previous and current action)
            current_total_wait = self._collect_waiting_times()

            reward = old_total_wait - current_total_wait

            if self._step != 0:
                # add this state/action/reward to replay memory
                self._replay_memory.add_sample((old_state, old_action, reward, current_state))

            # choose next action to take
            action = self._choose_action(current_state, epsilon)

            # if chosen light phase is a change, activate yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # save variables
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            if reward > 0:
                self._sum_total_reward += reward

        self._save_episode_stats()
        print("Total reward gained:", self._sum_total_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("TRAINING")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            # perform replay memory training on the neural net, one (batch) for each epoch selected
            self._replay()
        training_time = round(timeit.default_timer() - start_time,1)

        return simulation_time, training_time
        
    def _replay(self):
        batch = self._replay_memory.get_samples(self._neural_net.batch_size)

        # if samples are available, extract state and next state from each
        if len(batch) > 0:
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])

            qsa = self._neural_net.predict_batch(states)
            qsa_next = self._neural_net.predict_batch(next_states)

            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract state/action/reward/state'
                current_q = qsa[i]  # get the predicted Q(state)
                current_q[action] = reward + self._gamma * np.amax(qsa_next[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._neural_net.train_batch(x, y)  # train the NN



                



    def _choose_action(self, state, epsilon):
        # expoloration vs exploitation
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._neural_net.predict_single(state)) # the best action given the current state




    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()
        
        if networkID == 0:
            for car_id in car_list:
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 160:
                    lane_cell = 7
                elif lane_pos < 400:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell = 9

                # finding the lane where the car is located 
                # x2TL_3 are the "turn left only" lanes
                if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                    lane_group = 0
                elif lane_id == "W2TL_3":
                    lane_group = 1
                elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                    lane_group = 2
                elif lane_id == "N2TL_3":
                    lane_group = 3
                elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                    lane_group = 4
                elif lane_id == "E2TL_3":
                    lane_group = 5
                elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                    lane_group = 6
                elif lane_id == "S2TL_3":
                    lane_group = 7
                else:
                    lane_group = -1

                if lane_group >= 1 and lane_group <= 7:
                    car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        elif networkID == 1:
            for car_id in car_list:
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 160:
                    lane_cell = 7
                elif lane_pos < 400:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell = 9

                if lane_id == "E3":
                    lane_group = 0
                elif lane_id == "E5":
                    lane_group = 1
                elif lane_id == "E4":
                    lane_group = 2
                elif lane_id == "E6":
                    lane_group = 3
                else:
                    lane_group = -1

                if lane_group >= 1 and lane_group <= 3:
                    car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-39
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"
        
        elif networkID == 2:
            for car_id in car_list:
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 160:
                    lane_cell = 7
                elif lane_pos < 400:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell = 9

                if lane_id == "-E0_0":
                    lane_group = 0
                elif lane_id == "-E1_0":
                    lane_group = 1
                elif lane_id == "-E2_0":
                    lane_group = 2
                elif lane_id == "-E3_0":
                    lane_group = 3
                else:
                    lane_group = -1

                if lane_group >= 1 and lane_group <= 3:
                    car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-39
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        elif networkID == 3:
            for car_id in car_list:
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

                # distance in meters from the traffic light -> mapping into cells
                if lane_pos < 7:
                    lane_cell = 0
                elif lane_pos < 14:
                    lane_cell = 1
                elif lane_pos < 21:
                    lane_cell = 2
                elif lane_pos < 28:
                    lane_cell = 3
                elif lane_pos < 40:
                    lane_cell = 4
                elif lane_pos < 60:
                    lane_cell = 5
                elif lane_pos < 100:
                    lane_cell = 6
                elif lane_pos < 160:
                    lane_cell = 7
                elif lane_pos < 400:
                    lane_cell = 8
                elif lane_pos <= 750:
                    lane_cell = 9

                if lane_id == "E9":
                    lane_group = 0
                elif lane_id == "E10":
                    lane_group = 1
                elif lane_id == "E11":
                    lane_group = 2
                elif lane_id == "E8":
                    lane_group = 3
                else:
                    lane_group = -1

                if lane_group >= 1 and lane_group <= 3:
                    car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-39
                    valid_car = True
                elif lane_group == 0:
                    car_position = lane_cell
                    valid_car = True
                else:
                    valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

                if valid_car:
                    state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        #print(state)
        return state
    

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        if networkID == 0:
            incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        elif networkID == 1:
            incoming_roads = ["E3", "E5", "E4", "E6"]
        elif networkID == 2:
            incoming_roads = ["-E0", "-E1", "-E2", "-E3"]
        elif networkID == 3:
            incoming_roads = ["E9", "E10", "E11", "E8"]

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id] 
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        if networkID == 0:
            yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
            traci.trafficlight.setPhase("TL", yellow_phase_code)
        elif networkID == 1:
            #yellow_phase_code = old_action + 1 # obtain the yellow phase code, based on the old action (ref on simple-intersection.net.xml)
            if old_action == 0:
                yellow_phase_code = 1
            else:
                yellow_phase_code = 3
            traci.trafficlight.setPhase("J5", yellow_phase_code)
        elif networkID == 2:

            if old_action == 0:
                yellow_phase_code = 1
            else:
                yellow_phase_code = 4
            traci.trafficlight.setPhase("J6", yellow_phase_code)
            traci.trafficlight.setPhase("J8", yellow_phase_code)
            traci.trafficlight.setPhase("J7", yellow_phase_code)
            traci.trafficlight.setPhase("J9", yellow_phase_code)

        elif networkID == 3:
            if old_action == 0:
                yellow_phase_code = 1
            else:
                yellow_phase_code = 3
            traci.trafficlight.setPhase("J20", yellow_phase_code)
            traci.trafficlight.setPhase("J21", yellow_phase_code)
            traci.trafficlight.setPhase("J22", yellow_phase_code)
            traci.trafficlight.setPhase("J23", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if networkID == 0:
            if action_number == 0:
                traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
            elif action_number == 1:
                traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
            elif action_number == 2:
                traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
            elif action_number == 3:
                traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

        elif networkID == 1:
            if action_number == 0:
                traci.trafficlight.setPhase("J5", PHASE_NS_GREEN)
            elif action_number == 1:
                traci.trafficlight.setPhase("J5", PHASE_EW_GREEN)

        elif networkID == 2:
            if action_number == 0:
                traci.trafficlight.setPhase("J6", PHASE_NS_GREEN)
                traci.trafficlight.setPhase("J8", PHASE_NS_GREEN)
                traci.trafficlight.setPhase("J9", PHASE_NS_GREEN)
                traci.trafficlight.setPhase("J7", PHASE_NS_GREEN)
            elif action_number == 1:
                traci.trafficlight.setPhase("J6", PHASE_NS_THROUGH)
                traci.trafficlight.setPhase("J8", PHASE_NS_THROUGH)
                traci.trafficlight.setPhase("J9", PHASE_NS_THROUGH)
                traci.trafficlight.setPhase("J7", PHASE_NS_THROUGH)

        elif networkID == 3:
            if action_number == 0:
                traci.trafficlight.setPhase("J20", PHASE_NS_GREEN)
                traci.trafficlight.setPhase("J21", PHASE_NS_GREEN)
                traci.trafficlight.setPhase("J22", PHASE_NS_GREEN)
                traci.trafficlight.setPhase("J23", PHASE_NS_GREEN)
            if action_number == 1:
                traci.trafficlight.setPhase("J20", PHASE_EW_GREEN)
                traci.trafficlight.setPhase("J21", PHASE_EW_GREEN)
                traci.trafficlight.setPhase("J22", PHASE_EW_GREEN)
                traci.trafficlight.setPhase("J23", PHASE_EW_GREEN)
    
    def _simulate(self, steps_todo):
        # Execute steps in sumo
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1
            steps_todo -= 1
            new_queue_length = self._get_queue_length()
            self._sum_queue_length += new_queue_length
            # queue_length == waited_seconds
            self._sum_waiting_time += new_queue_length

    def _save_episode_stats(self):
        self._reward_store.append(self._sum_total_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        if networkID == 0:
            halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
            halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
            halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
            halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
            queue_length = halt_N + halt_S + halt_E + halt_W

        elif networkID == 1:
            halt_W = traci.edge.getLastStepHaltingNumber("E3")
            halt_N = traci.edge.getLastStepHaltingNumber("E5")
            halt_E = traci.edge.getLastStepHaltingNumber("E4")
            halt_S = traci.edge.getLastStepHaltingNumber("E6")
            queue_length = halt_W + halt_N + halt_E + halt_S

        elif networkID == 2:
            halt_W = traci.edge.getLastStepHaltingNumber("-E0")
            halt_N = traci.edge.getLastStepHaltingNumber("-E1")
            halt_E = traci.edge.getLastStepHaltingNumber("-E2")
            halt_S = traci.edge.getLastStepHaltingNumber("-E3")
            queue_length = halt_W + halt_N + halt_E + halt_S

        elif networkID == 3:
            halt_W = traci.edge.getLastStepHaltingNumber("E9")
            halt_N = traci.edge.getLastStepHaltingNumber("E10")
            halt_E = traci.edge.getLastStepHaltingNumber("E11")
            halt_S = traci.edge.getLastStepHaltingNumber("E8")
            queue_length = halt_W + halt_N + halt_E + halt_S

        return queue_length
    
    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store


    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
