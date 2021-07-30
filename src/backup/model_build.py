from os import path
import sys, json, time
from gymenv import make_env
import numpy as np
from policy_model import GAT
import config
import torch

# set path and input model_build input data
def input_data():
    # Get Directory
    BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

    training_routes_path = path.join(BASE_DIR, 'data/simple_data/simple_route_data.json')
    training_packages_path = path.join(BASE_DIR, 'data/simple_data/simple_package_data.json')
    training_travel_times_path = path.join(BASE_DIR, 'data/simple_data/simple_travel_times.json')
    training_actual_sequences_path = path.join(BASE_DIR, 'data/simple_data/actual_sequence.json')
    with open(training_routes_path, newline='') as in_file:
        actual_routes = json.load(in_file)

    with open(training_packages_path, newline='') as in_file:
        actual_packages = json.load(in_file)

    with open(training_travel_times_path, newline='') as in_file:
        actual_travel_times = json.load(in_file)

    with open(training_actual_sequences_path, newline='') as in_file:
        actual_sequences = json.load(in_file)

    data = [actual_routes, actual_packages, actual_travel_times, actual_sequences]
    return data


# change the state format into the GNN input format
def state2input(state, original_data):
    """
    Input:
    state: a compiled list
        include routeID, feature vectors, adjacent matrix
    Output:
    norm_x: [N, d] matrix, N = len(list(feature_dict)) + 1
        normalized feature matrix, N is the number of nodes, d is the number of features
    norm_adj: [N, N] matrix
        normalized adjacent matrix of the graph
    stops_table: [N] list
        use it to check the index of each stop
    """
    actual_routes, actual_packages, actual_travel_times, actual_sequences = original_data
    routeID, seq, departure_time, departure_stop, feature_dict = state
    N = len(list(actual_routes[routeID]['stops']))

    # create stop table
    stops_table = []
    for stop in feature_dict.keys():
        stops_table.append(stop)  # append unvisited stops

    # get adjacent matrix adj
    adj = np.zeros([N, N])
    travel_times = actual_travel_times[routeID]
    for i in range(N):
        for j in range(N):
            if i != j:
                start = list(actual_routes[routeID]['stops'])[i]
                end = list(actual_routes[routeID]['stops'])[j]
                adj[i, j] = travel_times[start][end]
            else:
                adj[i, j] = 0

    # get the feature vectors
    x = np.zeros([N, config.nfeat])
    for i in range(N):
        stop = list(actual_routes[routeID]['stops'])[i]
        # for visited stops
        if stop not in stops_table:
            stop_lat = actual_routes[routeID]['stops'][stop]['lat']
            stop_lng = actual_routes[routeID]['stops'][stop]['lng']
            add_feature = np.array([0, 0, 0, -999999, +999999, stop_lat, stop_lng])
            x[i, :] = add_feature
        # for unvisited stops
        else:
            x[i, :] = feature_dict[stop]

    # get the normalized x, adj
    norm_x, norm_adj = features_normalization(x, adj, N)

    return norm_x, norm_adj, stops_table


# normalize the x and adj matrices
def features_normalization(x, adj, N):
    norm_x = x.copy()
    norm_adj = adj.copy()

    # get norm_x by each column
    for j in range(config.nfeat):
        col = x[:, j]
        mean = np.mean(col)
        std = np.std(col) + 0.000001  # robustness
        min_value = np.inf

        # normalize
        for i in range(N):
            norm_x[i, j] = (x[i, j] - mean) / std
            if norm_x[i, j] < min_value:
                min_value = norm_x[i, j]
        # shift
        for i in range(N):
            norm_x[i, j] = norm_x[i, j] - min_value

    # get norm_adj
    time_list = []
    for i in range(N):
        for j in range(N):
            time_list.append(adj[i, j])
    time_mean = np.mean(time_list)
    time_std = np.std(time_list) + 0.000001  # robustness

    # normalize
    min_time = np.inf
    for i in range(N):
        for j in range(N):
            norm_adj[i][j] = (adj[i][j] - time_mean) / time_std
            if norm_adj[i][j] < min_time:
                min_time = norm_adj[i][j]
    # shift
    for i in range(N):
        for j in range(N):
            norm_adj[i][j] = norm_adj[i][j] - min_time

    return norm_x, norm_adj


if __name__ == "__main__":

    # create env
    original_data = input_data()
    actual_routes, actual_packages, actual_travel_times, actual_sequences = original_data
    env = make_env(config.train_num, original_data)

    # initialize GAT
    network = GAT(config.nfeat, config.nhid, config.dropout, config.alpha, config.lr)

    for ite in range(config.iterations):
        """
        observation: [norm_x, norm_adj]
            the input of the GNN
        action: int
            the index of the chosen stop
        values: float
            the Monte carlo value prediction, calculated based on instant reward
        reward: float
            the instant reward
        """
        obs = []  # states

        # gym reset
        done = False
        t = 0
        state = env.reset()  # sample a random instance
        routeID = state[0]
        N = len(list(actual_routes[routeID]['stops']))

        predict_list = []

        while not done:

            routeID, seq, departure_time, departure_stop, feature_dict = state

            # change the state format into the input format
            norm_x, norm_adj, stops_table = state2input(state, original_data)

            # select action according to the prob. distribution
            # ATTENTION: prob is actually the score of each node, because we omit the soft-max in the GNN
            prob = network.compute_prob(norm_x, norm_adj)

            # print prob for testing
            _prob = torch.from_numpy(prob).float()
            # print(torch.softmax(_prob, 0).cpu().data.numpy())

            # choose the stop with the highest score as the next stop
            stop_index = np.argmax(prob)
            min_prob = np.min(prob)
            action = list(actual_routes[routeID]['stops'])[stop_index]
            while action not in stops_table:
                prob[stop_index] = min_prob - 1
                stop_index = np.argmax(prob)
                action = list(actual_routes[routeID]['stops'])[stop_index]

            """
            # random action
            index = random.randint(0, N-1)
            action = list(actual_routes[routeID]['stops'])[index]
            while action not in stops_table:
                index = random.randint(0, N - 1)
                action = list(actual_routes[routeID]['stops'])[index]
            
            """

            # next state
            next_state, done = env.step(state, action)
            # print('iteration: ', ite, 'step: ', t, 'action: ', action, 'done: ', done)

            # append
            observation = [norm_x, norm_adj]
            obs.append(observation)
            predict_list.append(action)

            t += 1
            state = next_state

        # get the actual sequence
        target_list = []
        target = np.zeros([N - 1])  # do not need the initial stop
        stop_list = list(actual_routes[routeID]['stops'])
        key_list = list(actual_sequences[routeID]['actual'].keys())
        val_list = list(actual_sequences[routeID]['actual'].values())

        for i in range(1, N):
            position = val_list.index(i)
            stop = key_list[position]  # the i-th stop
            target[i - 1] = stop_list.index(stop)
            target_list.append(stop)

        network.Train(obs, target)

        print(stop_list)
        print('The predicted seq: ', predict_list)
        print('The actual seq: ', target_list)
        print('')

"""
    # Write output data
    model_path=path.join(BASE_DIR, 'data/model_build_outputs/model.json')
    with open(model_path, 'w') as out_file:
        json.dump(output, out_file)
        print("Success: The '{}' file has been saved".format(model_path))

"""
