import os

import numpy as np
from .cgp import FunctionGen


def plot_graph(individual, filename=None, rankdir='TB'):
    try:
        import pydot
        graph = pydot.Dot(graph_type='graph', rankdir=rankdir)
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
                fnc = individual.config.functions[n.fnc_idx]

                if isinstance(fnc, str):
                    name = fnc
                elif hasattr(fnc, 'name'):
                    name = fnc.name
                else:
                    name = fnc.__name__

                label = name
                name += '_id_%d' % idx
            else:
                name = 'output-%d' % out_idx
                label = name
                out_idx += 1

            nodes[idx + individual.config.num_input] = pydot.Node(name, label=label)

        for idx in active_nodes:
            node = nodes[idx + individual.config.num_input]
            for con in range(individual.genes[idx].num_inputs):
                con_node = nodes[individual.genes[idx].inputs[con]]
                graph.add_node(con_node)
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
        node = pydot.Node('output %d' % idx, style="filled", fillcolor='#ccaadd')
        node.set('pos', '%2f,%2f!' % ((individual.config.cols + 1) / 1.5, idx))
        nodes.append(node)
        graph.add_node(node)

    for idx, node in enumerate(nodes[individual.config.num_input:]):
        if not individual.active[idx]:
            continue

        for con in range(individual.genes[idx].num_inputs):
            graph.add_edge(pydot.Edge(nodes[individual.genes[idx].inputs[con]], node))

    return graph

