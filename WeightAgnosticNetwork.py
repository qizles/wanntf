from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
import gym
from gym_cartpole_swingup.envs import CartPoleSwingUpEnv

from MLPModel import MLP



BIAS_VALUE = 1.0
goal_steps = 500
score_requirement = 60
intial_games = 10000


def customStepFunction(value):
    # todo : make float tensor out of reuturn
    if value < 0:
        return -1
    return 1


def gaussian(x):
    return np.exp(-x ** 2)


def customAbsolute(x):
    return np.abs(x)

def customInvert(x):
    return -x

activations_dict = {'linear': tf.keras.activations.linear, 'step': customStepFunction,
                    'sin': tf.math.sin, 'cosine': tf.math.cos,
                    'Gaussian': gaussian, 'tanh': tf.keras.activations.tanh,
                    'sigmoid': tf.keras.activations.sigmoid, 'absolute value': customAbsolute,
                    'invert': customInvert, 'ReLU': tf.keras.activations.relu}




class CustomModel(Model):
    def __init__(self, model_plan, shared_weight=None):
        super(CustomModel, self).__init__()
        self.custom_layers = []

        self.input_node_ids = [elem for elem in model_plan.layers[0].nodes]




        for layer in model_plan.layers[1:]:  # for all layers except input layer
            self.custom_layers.append(CustomLayer(layer.nodes, model_plan.connections, shared_weight))

    @property
    def last_layer(self):
        return self.custom_layers[-1]

    def changeSharedWeight(self, shared_weight):
        for layer in self.custom_layers:
            layer.changeSharedWeight(shared_weight)

    def printWeights(self):
        print("print weights of tf model")


        weights = []
        for layer in self.custom_layers:
            weights.append(layer.getWeights())
        print(weights)

    def call(self, inputs, training=None, mask=None):

        inputs = dict(zip(self.input_node_ids, inputs))
        for layer in self.custom_layers:

            inputs = layer(inputs)
        output = []
        for node in self.last_layer.nodes:
            output.append(inputs[node])

        return output


class CustomLayer(layers.Layer):
    def __init__(self, neurons, connections, shared_weight=None):
        super(CustomLayer, self).__init__()

        self.nodes = {}
        self.act_funcs = {}

        # neurons that shall be in the layer
        for neuron in neurons:
            # should get all connections coming into a single neuron
            neuron_connections = [item for item in connections if item[1] == neuron]
            # print("neurcon connections")
            # print(neuron_connections)
            # neuron_connections = connections.get(neuron.id)

            # now we need to add the connections to our layer
            curr_connections = {}
            for nc in neuron_connections:
                if shared_weight:
                    curr_connections[nc[0]] = tf.Variable(shared_weight, dtype=tf.float64)
                else:
                    print("should not occur")
                    exit(1)
                    curr_connections[nc[0]] = tf.Variable(initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None), dtype=tf.float64)
            self.nodes[neuron] = curr_connections

            # self.act_funcs[neuron] = tf.keras.activations.deserialize(neuron.activation)
            self.act_funcs[neuron] = activations_dict[neuron.activation]

            # print("custom layer created")

    def changeSharedWeight(self, shared_weight):
        for node, connections in self.nodes.items():
            for nc, weight in connections.items():
                connections[nc] = tf.Variable(shared_weight, dtype=tf.float64)

    def getWeights(self):
        weights = []
        for node, connections in self.nodes.items():
            for nc, weight in connections.items():
                weights.append(weight)
        return weights


    def call(self, inputs, **kwargs):
        weights_list = []
        for node, connections in self.nodes.items():
            value = tf.Variable(0, dtype=tf.float64)
            #test if neuron has multiple connections
            for nc, weight in connections.items():
                if inputs.get(nc).numpy() == 1 or inputs.get(nc).numpy() == -1:

                    value.assign_add(tf.cast(inputs.get(nc), dtype=tf.float64))

                else:
                    value.assign_add(inputs.get(nc) * weight)
                weights_list.append(weight)
            value = self.act_funcs[node](value)
            inputs[node] = value
        # print("weights")
        # print(weights_list)
        return inputs

