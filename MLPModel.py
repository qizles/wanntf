import numpy as np



kreas_activation_functions = 'elu', 'exponential', 'hard_sigmoid', 'linear', 'relu', 'selu', 'sigmoid', 'softplus',\
                             'softmax', 'softsign', 'tanh'

# written as in the paper
activationFunctions = 'linear', 'step', 'sin', 'cosine', 'Gaussian', 'tanh', 'sigmoid', 'absolute value', 'invert',\
                      'ReLU'


def getActFunc(oldFunction = None):
    while True:
        function = activationFunctions[np.random.randint(0, len(activationFunctions))]
        if not function == oldFunction:
            break
    return function


class Neuron:
    def __init__(self, layer, activation_function=None):
        #
        # self.connections = []
        if activation_function:
            self.activation = activation_function
        else:
            self.activation = getActFunc()
        self.layer = layer

    def layerRank(self, layers):
        return layers.index(self.layer)

    #def __hash__(self):
    #   return id(self)

    def __lt__(self, other):
        return id(self) < id(other)


class Layer:
    def __init__(self):
        self.nodes = []


class MLP:
    def __init__(self, n_inputs, n_outputs, n_connections):
        self.layers = []
        self.connections = []


        input_layer = Layer()
        self.layers.append(input_layer)
        for _ in range(n_inputs + 1):  # +1 for bias
            input_layer.nodes.append(Neuron(input_layer, 'linear'))

        output_layer = Layer()
        self.layers.append(output_layer)
        for _ in range(n_outputs):
            output_layer.nodes.append(Neuron(output_layer, 'linear'))
        for _ in range(n_connections):
            self.addConnection()


    def __str__(self):
        print("------------------------------")
        for layer in self.layers:
            print('layer {}'.format(layer))
            for neuron in layer.nodes:
                print('neuron {}'.format(neuron))
                for connection in self.connections:
                    if connection[1] == neuron:
                        print("connection from {}".format(connection[0]))
        return "----------------------------"

    @property
    def complexity(self):
        return len(self.connections)








    @property
    def neurons(self):
        neurons = []
        for layer in self.layers:
            for neuron in layer.nodes:
               neurons.append(neuron)
        return neurons



    def insertNode(self):
        print("insert node in modelplan")
        random = np.random.randint(0, len(self.connections))
        print(random)
        print(len(self.connections))
        connection = self.connections[random]
        if abs(connection[1].layerRank(self.layers) - connection[0].layerRank(self.layers)) == 1:
            layer = Layer()
            print("insert in new layer")
            self.layers.insert(connection[0].layerRank(self.layers) + 1, layer)
        else:
            print('insert in old layer')
            layer = self.layers[connection[0].layerRank(self.layers) + 1]

        newNode = Neuron(layer)
        self.connections.remove(connection)
        self.connections.append((connection[0], newNode))
        self.connections.append((newNode, connection[1]))
        layer.nodes.append(newNode)

    def addConnection(self):
        neuron1 = self.neurons[np.random.randint(0, len(self.neurons))] # todo : why infinite loop when only choosing second partner in loop ?
        print("add conncection to modelplan")
        max_tries = 20
        tries = 0
        while True:
            print("looking for connection partners")
            neuron1 = self.neurons[np.random.randint(0, len(self.neurons))]
            neuron2 = self.neurons[np.random.randint(0, len(self.neurons))]
            if not neuron1 == neuron2 and not neuron1.layerRank(self.layers) == neuron2.layerRank(self.layers):
                if neuron1.layerRank(self.layers) > neuron2.layerRank(self.layers):
                    neuronTemp = neuron2
                    neuron2 = neuron1
                    neuron1 = neuronTemp
                if (neuron1, neuron2) not in self.connections:
                    self.connections.append((neuron1, neuron2))
                    return
            tries += 1
            if tries > max_tries:
                print("No possible candidate to add connection")
                return

    def changeActivation(self):
        print("change activation of modelplan")
        max_tries = 20
        tries = 0
        while True:
            neuron = self.neurons[np.random.randint(0, len(self.neurons))]
            if not neuron.layer == self.layers[0] and not neuron.layer == self.layers[-1]:
                break
            tries += 1
            if tries > max_tries:
                print("No possible candidate for activation change")
                return
        print('changed activation')
        neuron.activation = getActFunc(neuron.activation)

