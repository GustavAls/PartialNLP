import os.path

from matplotlib import pyplot
from math import cos, sin, atan
import numpy as np
import matplotlib as plt

class Neuron():
    def __init__(self, x, y, is_stochastic = False):
        self.x = x
        self.y = y
        self.is_stochastic = is_stochastic

    def draw(self, neuron_radius, last_layer = 0):
        if self.is_stochastic:
            color = 'tab:green'
        else:
            color = 'tab:gray'
        if last_layer == -1:
            color = 'white'
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False, color=color, linewidth=13)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.horizontal_distance_between_middle_neurons = 1
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)
        self.__init_stochastic_neurons()
    def __init_stochastic_neurons(self):

        indices = np.random.choice(np.arange(0, len(self.neurons)), size = (len(self.neurons)//2, ),replace=False)
        if len(indices) > 0:
            for ind in indices:
                setattr(self.neurons[ind], 'is_stochastic', True)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, last_layer = 0):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        if neuron1.is_stochastic and neuron2.is_stochastic:
            color = 'tab:green'
            alpha = 1
            linewidth = 13
        else:
            color = 'tab:gray'
            alpha = 0.5
            linewidth = 13
        if last_layer==-1:
            color = 'white'
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
                             color = color, alpha=alpha, linewidth=linewidth)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        # if layerType == 0:
        #     pyplot.text(x_text, self.y, 'Input Layer', fontsize = 50)
        # elif layerType == -1:
        #     pyplot.text(x_text, self.y, 'Output Layer', fontsize = 50)
        # else:
        #     pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 50)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons ):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw(self,save_path = None):
        pyplot.figure(figsize=(50, 20))
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        # pyplot.title('Key', fontsize=50 )
        if save_path is not None:
            pyplot.savefig(save_path, format='png')
        pyplot.show()

class DrawNN():
    def __init__( self, neural_network ):
        self.neural_network = neural_network

    def draw( self, save_path = None):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer )
        for l in self.neural_network:
            network.add_layer(l)

        network.draw(save_path)



if __name__ == '__main__':
    main_path = r'C:\Users\45292\Documents\Master\SentimentClassification\IMDB Figures\Model General'
    for t in ['out_lin']:
        save_path = os.path.join(main_path, f'{t}.png')
        network = DrawNN([8, 16, 8])
        network.draw(save_path)