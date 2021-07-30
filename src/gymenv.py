import numpy as np
from os import path
import json
import random
import pandas as pd
from score import score

"""
state
routeID: str 
    the route id of this trip
seq: dict
    visited stops, included departure stop
departure_time: pd.datetime
    the time when drive leaves the departure stop
departure_stop: str
    the stop that the drive leaves
feature_dict: dict
    include all unvisited stops' feature vectors

stop_chosen: str 
    next stop
"""


class Env(object):
    def __init__(self, routes, original_data):
        assert len(routes) > 0
        self.routes = routes  # the list of routesID used to train
        self._routes, self._packages, self._travel_times, self._sequences = original_data
        self.all_indices = list(range(len(self.routes)))
        self.available_indices = list(range(len(self.routes)))
        self.env_index = None

    # get a new routeID from the training set
    def reset(self):
        self.env_index = np.random.choice(self.available_indices)
        self.available_indices.remove(self.env_index)
        if len(self.available_indices) == 0:
            self.available_indices = self.all_indices[:]

        select_routeID = self.routes[self.env_index]  # the selected route ID
        initial_state = self.get_initial_state(select_routeID)
        return initial_state

    # get the next_state, reward, done based on current_state and action
    def step(self, state, action):
        """
        state
        routeID: str
            the route id of this trip
        seq: dict
            visited stops, included departure stop
        departure_time: pd.datetime
            the time when drive leaves the departure stop
        departure_stop: str
            the stop that the drive leaves
        feature_dict: dict
            include all unvisited stops' feature vectors

        action: str
            next stop
        """
        routeID, seq, departure_time, departure_stop, feature_dict = state
        packages = self._packages[routeID].copy()

        # update seq
        num = len(list(seq))
        seq[action] = num  # add the next stop into sequence

        # update the departure_time
        travel_time = self._travel_times[routeID][departure_stop][action]
        service_time = self.combine_service_time(packages, departure_stop)
        spent_time = travel_time + service_time
        departure_time = departure_time + pd.to_timedelta(spent_time, unit='s')

        # update the next feature_dict
        feature_dict.pop(action)  # only contain unvisited stops
        for feature in feature_dict.values():
            feature[3] = feature[3] - spent_time  # time_till_start
            feature[4] = feature[4] - spent_time  # time_till_end

        # update departure_stop
        departure_stop = action

        # update done
        if len(list(feature_dict)) == 0:
            done = True
        else:
            done = False

        next_state = [routeID, seq, departure_time, departure_stop, feature_dict]

        return next_state, done

    # get the score of the partial solution (compare actual sequence and completed sequence)
    # This function was used in RL, we do not need it anymore.
    """
    def score_Cal(self, routeID, seq):
        actual_seq = self._sequences[routeID]['actual'].copy()
        complete_seq = self.complete_seq(seq, actual_seq)
        cost_mat = self._travel_times[routeID].copy()

        assert len(list(actual_seq)) == len(list(complete_seq))

        actual_list = self.route2list(actual_seq)
        complete_list = self.route2list(complete_seq)

        # use the score function provided from rc-cli
        return score(actual_list, complete_list, cost_mat)
    """

    # get the initial state after reset
    def get_initial_state(self, routeID):
        """
        state
        routeID: str
            the route id of this trip
        seq: dict
            visited stops, included departure stop
        departure_time: pd.datetime
            the time when drive leaves the departure stop
        departure_stop: str
            the stop that the drive leaves
        feature_dict: dict
            include all unvisited stops' feature vectors

        action: str
            next stop
        """
        route = self._routes[routeID].copy()
        packages = self._packages[routeID].copy()
        stops = self._routes[routeID]['stops'].copy()

        # get the departure time,
        departure_day = route['date_YYYY_MM_DD']
        departure_hour = route['departure_time_utc']
        departure_time = departure_day + " " + departure_hour
        departure_time = pd.to_datetime(departure_time, format='%Y-%m-%d %H:%M:%S')

        # get the departure stop
        departure_stop = None
        for key, value in stops.items():
            if value['type'] == 'Station':
                departure_stop = key

        # get the stop feature vectors
        list_stop_features = []
        assert departure_stop is not None
        stops.pop(departure_stop)

        for stop in stops.keys():
            stop_feature = self.get_feature(stops, packages, stop, departure_time)
            list_stop_features.append((stop, stop_feature))
        feature_dict = dict(list_stop_features)

        # return the initial state
        seq = {departure_stop: 0}
        state = [routeID, seq, departure_time, departure_stop, feature_dict]
        return state

    # return the feature vector in this stop
    def get_feature(self, stops, packages, stop, departure_time):
        stop_features = []

        # get the number of packages in this stop
        package_num = len(list(packages[stop]))

        # get the total volume in this stop
        total_volume = self.combine_volume(packages, stop)

        # get the total service time in this stop
        total_service_time = self.combine_service_time(packages, stop)

        # get the combined start time and end time in this stop
        if self.combine_date(packages, stop) is not False:
            final_start_time, final_end_time = self.combine_date(packages, stop)
            time_till_start = (final_start_time - departure_time).seconds  # unit: seconds
            time_till_end = (final_end_time - departure_time).seconds  # unit: seconds
        else:
            time_till_start = -999999  # no constraint on time window
            time_till_end = 999999  # no constraint on time window

        # get the latitude and longitude of this place
        latitude = stops[stop]['lat']
        longitude = stops[stop]['lng']

        stop_features.append(package_num)
        stop_features.append(total_volume)
        stop_features.append(total_service_time)
        stop_features.append(time_till_start)
        stop_features.append(time_till_end)
        stop_features.append(latitude)
        stop_features.append(longitude)

        return np.array(stop_features)

    # return the latest start time and the earliest end time
    @staticmethod
    def combine_date(packages, stop):
        packages = packages[stop]  # all packages in this stop
        start_times_list = []
        end_times_list = []

        for item in packages.values():
            start_time = item['time_window']['start_time_utc']
            end_time = item['time_window']['end_time_utc']

            if start_time == start_time:  # if not, it is nan
                start_time = pd.to_datetime(start_time, format='%Y-%m-%d %H:%M:%S')
                start_times_list.append(start_time)

            if end_time == end_time:  # if not, it is nan
                end_time = pd.to_datetime(end_time, format='%Y-%m-%d %H:%M:%S')
                end_times_list.append(end_time)

        if len(start_times_list) > 0 and len(end_times_list) > 0:
            final_start_time = max(start_times_list)
            final_end_time = min(end_times_list)
            return final_start_time, final_end_time
        else:
            return False

    # return the total volume
    @staticmethod
    def combine_volume(packages, stop):
        packages = packages[stop]  # all packages in this stop
        total_volume = 0
        for item in packages.values():
            dim = item['dimensions']
            total_volume += dim['depth_cm'] * dim['height_cm'] * dim['width_cm']

        return total_volume

    # return the total service time
    @staticmethod
    def combine_service_time(packages, stop):
        packages = packages[stop]  # all packages in this stop
        total_service_time = 0
        for item in packages.values():
            total_service_time += item['planned_service_time_seconds']

        return total_service_time

    # complete the sequence
    @staticmethod
    def complete_seq(seq, actual_seq):
        propose_Seq = seq.copy()
        actual_Seq = actual_seq.copy()
        propose_len = len(list(propose_Seq))
        actual_len = len(list(actual_Seq))

        # pop the visited stops
        for stop in propose_Seq.keys():
            actual_Seq.pop(stop)

        # sort the unvisited stops
        sort_actual_Seq = {k: v for k, v in sorted(actual_Seq.items(), key=lambda item: item[1])}

        # complete the propose sequence
        for i in range(propose_len, actual_len):
            stop = list(sort_actual_Seq)[i - propose_len]
            propose_Seq[stop] = i

        return propose_Seq

    # route2list
    @staticmethod
    def route2list(stops):
        route_list = [0] * (len(stops) + 1)
        for stop in stops:
            route_list[stops[stop]] = stop
        route_list[-1] = route_list[0]
        return route_list


def make_env(train_num, original_data):
    # input data
    actual_routes, actual_packages, actual_travel_times, actual_sequences = original_data

    # randomly pick routeID
    routes = []
    for i in range(train_num):
        route_ID = random.choice(list(actual_routes))  # choose a random route for training
        routes.append(route_ID)  # append high quality routeID

    return Env(routes, original_data)
