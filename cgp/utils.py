import os
from types import NoneType

import numpy as np
from .cgp import FunctionGen, Individual

def plot_graph(individual, filename=None):
    try:
        import pydot
        graph = pydot.Dot(graph_type='graph')
        active_nodes = np.where(individual.active)[0]

        nodes = {}
        for i in range(individual.config.num_input):
            node = pydot.Node('input-%d' % i)
            nodes[i] = node
            graph.add_node(node)

        out_idx = 0
        for idx in active_nodes:
            n = individual.genes[idx]

            if isinstance(n, FunctionGen):
                if hasattr(n.fnc, 'name'):
                    name = n.fnc.name
                else:
                    name = n.fnc.__name__

                name += '_id_%d' % idx
            else:
                name = 'output-%d' % out_idx
                out_idx += 1

            nodes[idx + individual.config.num_input] = pydot.Node(name)

        for idx in active_nodes:
            node = nodes[idx + individual.config.num_input]
            for con in range(individual.genes[idx].num_inputs):
                con_node = nodes[individual.genes[idx].inputs[con]]
                graph.add_edge(pydot.Edge(con_node, node))

        if filename is not None:
            graph.write_png(os.path.abspath(filename))

        return graph
    except ImportError:
        raise ImportError("pydot not found please install it with pip")

def plot_cartesian(individual, filename='grid.png'):
    import pydot
    graph = pydot.Dot(graph_type='digraph')
    nodes = []

    for i in range(individual.config.num_input):
        node = pydot.Node('input-%d' % i)
        node.set('pos', '-1,%2f!' % i)
        nodes.append(node)
        graph.add_node(node)

    for idx in range(individual.config.num_nodes):
        x = min(idx // individual.config.rows, individual.config.cols)
        y = idx % individual.config.rows

        node = pydot.Node(idx, style="filled", shape="circle")
        node.set('pos', '%f,%f!' % (x / 1.5, y / 1.5))
        node.set('fontsize', 10)

        if individual.active[idx]:
            node.set('fillcolor', '#ff00cc')

        nodes.append(node)
        graph.add_node(node)

    for idx in range(individual.config.num_output):
        node = pydot.Node('output %d' % (idx + 1), style="filled", fillcolor='#ccaadd')
        node.set('pos', '%2f,%2f!' % ((individual.config.cols + 1) / 1.5, idx))
        nodes.append(node)
        graph.add_node(node)

    for idx, node in enumerate(nodes[individual.config.num_input:]):
        if not individual.active[idx]:
            continue

        for con in range(individual.genes[idx].num_inputs):
            graph.add_edge(pydot.Edge(nodes[individual.genes[idx].inputs[con]], node))

    return graph

def individual_to_keras_model(individual, input_shape=(64,64,3)):
    if not isinstance(individual, Individual):
        raise TypeError("Individual must be the type Individual")

    plot_graph(individual, 'graph.png')

    from keras.layers import Input, MaxPool2D, Concatenate
    from keras.layers.merge import _Merge
    import keras.backend as K
    import tensorflow as tf

    active_nodes = np.where(individual.active)[0]

    nodes = {}
    for i in range(individual.config.num_input):
        node = Input(shape=input_shape, name='input-%d' % i)
        nodes[i] = node

    for idx in active_nodes:
        n = individual.genes[idx]

        if isinstance(n, FunctionGen):
            nodes[idx + individual.config.num_input] = n.fnc

    for idx in active_nodes:
        if idx >= individual.config.num_nodes:
            continue

        node = nodes[idx + individual.config.num_input]
        node.name += '_id_%d' % idx
        print("node: %s" % node.name)

        if isinstance(node, _Merge) and individual.genes[idx].num_inputs == 2:
            x = []
            shapes = []
            for con in range(individual.genes[idx].num_inputs):
                instance = nodes[individual.genes[idx].inputs[con]]
                x.append(instance)
                shapes.append(instance.shape.as_list())

            _, a_width, a_height, a_channels = shapes[0]
            _, b_width, b_height, b_channels = shapes[1]

            if a_width > b_width:
                x[0] = MaxPool2D(pool_size=(a_height // b_height, a_width // b_width))(x[0])
            if a_width < b_width:
                x[1] = MaxPool2D(pool_size=(b_height // a_height, b_width // a_width))(x[1])

            if a_channels > b_channels:
                diff = a_channels - b_channels
                x[1] = tf.pad(x[1], ((0,0),(0,0),(0,0),(0, diff)), mode='CONSTANT')

            elif a_channels < b_channels:
                diff = b_channels - a_channels
                x[0] = tf.pad(x[0], ((0, 0), (0, 0), (0, 0), (0, diff)), mode='CONSTANT')

            print("")

        elif individual.genes[idx].num_inputs == 1:
            x = nodes[individual.genes[idx].inputs[0]]
            #graph.add_edge(pydot.Edge(con_node, node))
        else:
            pass

        print("call %s with value of %s" % (node.name, x.name if not isinstance(x, list) else [i.name for i in x]))
        nodes[idx + individual.config.num_input] = node(x)

