import torch
from os import path
import json
from gymenv import make_env
import numpy as np
import config
from models.policy_model import Policy
from models.value_function_model import ValueFunction


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


# calculate the discounted rewards
def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)


if __name__ == "__main__":
    # import data and create env
    original_data = input_data()
    actual_routes, actual_packages, actual_travel_times, actual_sequences = original_data
    env = make_env(config.train_num, original_data)

    # initialize policy
    actor = Policy()
    baseline = ValueFunction()

    # main iteration
    for ite in range(config.iterations):
        OBS = []  # observations
        ACTS = []  # actions
        ADS = []  # advantages
        VALS = []    # Monte carlo value predictions

        for num in range(config.numtrajs):
            obss = []  # states
            acts = []  # actions
            rews = []  # instant rewards

            obs = env.reset()  # sample a random instance
            routeID = obs[0]
            N = len(list(actual_routes[routeID]['stops']))
            done = False

            while not done:
                routeID, seq, departure_time, departure_stop, feature_dict = obs

                # change the state format into the input format
                norm_x, norm_adj, stops_table = state2input(obs, original_data)

                # choose a unvisited stop as the next stop
                prob = actor.compute_prob(norm_x, norm_adj)
                action = np.asscalar(np.random.choice(len(prob), p=prob.flatten(), size=1))
                next_stop = list(actual_routes[routeID]['stops'])[action]
                while next_stop not in stops_table:
                    action = np.asscalar(np.random.choice(len(prob), p=prob.flatten(), size=1))
                    next_stop = list(actual_routes[routeID]['stops'])[action]

                # step
                next_obs, reward, done = env.step(obs, next_stop)

                # append
                obss.append([norm_x, norm_adj])  # append node feature and adjacent matrix
                acts.append(action)
                rews.append(reward)

                obs = next_obs

            # Collect numtrajs trajectories for batch update
            num_step = len(obss)
            vals = discounted_rewards(rews, config.gamma)
            for i in range(num_step):
                OBS.append(obss[i])
                ACTS.append(acts[i])
                VALS.append(vals[i])

            # train baseline
            baseline.train(OBS, VALS)

            # update policy
            for i in range(num_step):
                ADS.append(VALS[i] - baseline.compute_values(OBS[i]))

            actor.train(OBS, ACTS, ADS)



