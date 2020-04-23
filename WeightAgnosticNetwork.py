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
    if value < 0:
        return tf.Variable(-1, dtype=tf.float64)
    return tf.Variable(1, dtype=tf.float64)


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
        print("they are the same lists : {}".format(self.input_node_ids == model_plan.layers[0].nodes))
        self.input_node_ids = model_plan.layers[0].nodes if self.input_node_ids == model_plan.layers[0].nodes else self.input_node_ids


        for layer in model_plan.layers[1:]:  # for all layers except input layer
            self.custom_layers.append(CustomLayer(layer.nodes, model_plan.connections, shared_weight))

    @property
    def last_layer(self):
        return self.custom_layers[-1]

    def changeSharedWeight(self, shared_weight):
        for layer in self.custom_layers:
            layer.changeSharedWeight(shared_weight)

    def printWeights(self):
        self.__str__()

    def __str__(self):
        print("weights of tf model {}".format(self))
        weights = []
        for layer in self.custom_layers:
            weights.append(layer.getWeights())
        print(weights)
        return ""

    def call(self, inputs, training=None, mask=None):
        # create dictionary of inputs with corresponding ids of input layer
        inputs = dict(zip(self.input_node_ids, inputs))
        # and propagate the same dict through all layers
        for layer in self.custom_layers:
            inputs = layer(inputs)
        # return only outputs of last layer, internal structure is now unneccessary
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
            # get all connections coming into this neuron from the model plan
            neuron_connections = [item for item in connections if item[1] == neuron]


            # now we need to add the connections with weight to our neuron
            # we create a dictionary with the neuron from which the connection comes from as the key
            # and store the weight alongside
            curr_connections = {}

            for nc in neuron_connections:
                if shared_weight:
                    curr_connections[nc[0]] = tf.Variable(shared_weight, dtype=tf.float64)
                else:
                    curr_connections[nc[0]] = tf.Variable(initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None), dtype=tf.float64)
            self.nodes[neuron] = curr_connections

            # and define a activation function for every neuron in this layer
            self.act_funcs[neuron] = activations_dict[neuron.activation]


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
        # we look at every neuron in our layer
        for node, connections in self.nodes.items():
            value = tf.Variable(0, dtype=tf.float64)
            # and sum up every in going input
            for nc, weight in connections.items():
                if inputs.get(nc).numpy() == 1 or inputs.get(nc).numpy() == -1:

                    value.assign_add(tf.cast(inputs.get(nc), dtype=tf.float64))

                else:
                    value.assign_add(inputs.get(nc) * weight)
            # and use the combined value to create the output for this neuron
            # and put it to the previous output to make them accessible for following layers,
            value = self.act_funcs[node](value)
            inputs[node] = value
        return inputs

