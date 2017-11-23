import operator
from copy import deepcopy, copy
from threading import Thread
import pickle
import numpy as np


# based on https://github.com/sg-nm/cgp-cnn/blob/master/
# but for deeper details I can recommend  http://www.cartesiangp.co.uk/
# especially the free chapter 1 and 2 from the free the CGP book


class OutputGen:
    def __init__(self, max_inputs):
        self.is_output = True
        self.inputs = np.zeros(max_inputs, dtype=int)
        self.num_inputs = 1


class FunctionGen:
    def __init__(self, fnc, max_inputs, num_inputs):
        if not callable(fnc):
            raise TypeError("fnc must be callable")

        self.fnc = fnc
        self.inputs = np.zeros(max_inputs, dtype=int)
        self.num_inputs = num_inputs
        self.is_output = False

    def __str__(self):
        return "fnc: %s with %s inputs" % (self.fnc, self.inputs)


class CgpConfig:
    def __init__(self, rows=5, cols=10, level_back=5, functions=[], function_inputs=[],
                 constraints=None, mutation_rate=0.1):
        """
        configuration for a CGP problem
        Parameters
        ----------
        rows: int(5)
            number of rows of a cgp grid
        cols: int(10)
            number of cols of a cgp grid
        level_back: int(5)
            max level back
        functions: list(None)
            list of available functions for a function node
        function_inputs: list(None)
            list containing number of inputs for each available function
        constraints:
            not used at the moment
        mutation_rate: float(0.1)
            percentage value of how many genes will be mutating
        """
        if not isinstance(functions, list) or not isinstance(function_inputs, list):
            raise TypeError("functions and function_inputs must be a list")

        if not len(functions) == len(function_inputs):
            raise ValueError("function list must have the same dimension like function_inputs")

        self.rows = rows
        self.cols = cols
        self.level_back = level_back

        self.num_input = 1
        self.num_output = 1
        self.num_nodes = rows * cols
        self.functions = functions
        self.function_inputs = function_inputs
        self.num_func_genes = len(functions)
        self.max_inputs = np.max(function_inputs)
        self.contraints = constraints

        self.mutation_rate = mutation_rate


class Individual:
    def __init__(self, config):
        if not isinstance(config, CgpConfig):
            raise ValueError("invalid value for config")

        self.config = config
        self.score = None

        # init genes with a random function
        self.genes = []
        for node in range(config.num_nodes + config.num_output):
            if node < self.config.num_nodes:
                fnc_idx = np.random.randint(self.config.num_func_genes)
                gene = FunctionGen(copy(self.config.functions[fnc_idx]),
                                   self.config.max_inputs,
                                   self.config.function_inputs[fnc_idx])
            else:
                gene = OutputGen(self.config.max_inputs)

            col = min(node // self.config.rows, self.config.cols)
            max_con_id = col * self.config.rows + self.config.num_input
            min_con_id = 0

            if col - self.config.level_back >= 0:
                min_con_id = (col - self.config.level_back) * self.config.rows + self.config.num_input

            # randomly connect genes
            for i in range(len(gene.inputs)):
                n = min_con_id + np.random.randint(max_con_id - min_con_id)
                gene.inputs[i] = n

            self.genes.append(gene)

        # lookup table for active nodes
        self.active = np.empty(len(self.genes), dtype=bool)
        self.check_active()

    def __walk_to_out(self, node):
        if not self.active[node]:
            self.active[node] = True

            for i in range(self.genes[node].num_inputs):
                if self.genes[node].inputs[i] >= self.config.num_input:
                    self.__walk_to_out(self.genes[node].inputs[i] - self.config.num_input)

    def __mutate_function_gene(self, gene_idx):
        # if not self.active[gene_idx]:
        fnc_idx = np.random.randint(self.config.num_func_genes)
        while fnc_idx == self.config.functions.index(self.genes[gene_idx].fnc):
            fnc_idx = np.random.randint(self.config.num_func_genes)

        self.genes[gene_idx].fnc = self.config.functions[fnc_idx]
        self.genes[gene_idx].num_inputs = self.config.function_inputs[fnc_idx]

        return True

    def __mutate_connection_gene(self, gene_idx):
        col = min(gene_idx // self.config.rows, self.config.cols)
        max_con_id = col * self.config.rows + self.config.num_input
        min_con_id = 0

        if col - self.config.level_back >= 0:
            min_con_id = (col - self.config.level_back) * self.config.rows + self.config.num_input

        for i in range(self.genes[gene_idx].num_inputs):
            if np.random.randint(0, 2) and max_con_id - min_con_id > 1:
                n = min_con_id + np.random.randint(max_con_id - min_con_id)
                while n == self.genes[gene_idx].inputs[i]:
                    n = min_con_id + np.random.randint(max_con_id - min_con_id)
                self.genes[gene_idx].inputs[i] = n

                return True

        return False

    def mutate(self, force=True):
        """
        lets a gene to mutate

        Parameters
        ----------
        force: bool(True)
            forces to mutate. if nothing changed it tries to mutate unitl something changed

        Returns
        -------
        None
        """
        if force:
            old_net = self.active.copy()

        cnt = 0
        num_genes = len(self.genes)
        while cnt <= int(num_genes * self.config.mutation_rate):
            idx = np.random.randint(0, num_genes)

            if idx < self.config.num_nodes:
                cnt += self.__mutate_function_gene(idx)

            # mutate connections
            cnt += self.__mutate_connection_gene(idx)

        self.check_active()

        if force:
            if np.array_equal(self.active, old_net):
                self.mutate(force)

    def check_active(self):
        self.active[:] = False
        for node in range(self.config.num_output):
            self.__walk_to_out(self.config.num_nodes + node)

    def num_active_nodes(self):
        """
        counts number of active genes

        Returns
        -------
        int
        """
        return len(np.where(self.active)[0])

    def active_net(self):
        """
        creates a decoded net structure

        Returns
        -------

        """
        net = [["input %d" % i, i, i] for i in range(self.config.num_input)]


        active_cnt = np.arange(self.config.num_nodes + self.config.num_output + self.config.num_input)
        active_cnt[self.config.num_input:] = np.cumsum(self.active)

        out_idx = 0
        for node, active in enumerate(self.active):
            if not active:
                continue

            con = [active_cnt[self.genes[node].inputs[i]] for i in range(self.genes[node].num_inputs)]
            if hasattr(self.genes[node], 'fnc'):
                if hasattr(self.genes[node].fnc, 'name'):
                    name = self.genes[node].fnc.name
                else:
                    name = self.genes[node].fnc.__name__
                net.append([name + "_id_%d" % len(net)] + con)
            else:
                net.append(['output-%d' % out_idx] + con)
                out_idx += 1

        return net


class CGP:
    def __init__(self, config, children=2):
        """
        apply Cartesian Genetic Programming to your problem

        Parameters
        ----------
        config: CgpConfig
            configuration for cgp instance
        children: int(2)
            number of children for evolution strategy
        """
        if not isinstance(config, CgpConfig):
            raise TypeError("config must be an instance of CgpConfig!")

        # create 1 parent + N childrens generations
        self.config = config
        self.num_children = children

    def __evaluator_wrapper(self, evaluator, child, index):
        score = evaluator(child, index)
        child.score = score

    def run(self, evaluator, max_epochs=10, force_mutate=True, op=operator.gt, verbose=1):
        """
        starts evaluation and searching process for a problem

        Parameters
        ----------
        evaluator: function(child, child_index)
            function that will be called for each individual to rate it
            arguments are individual and an index
            the function must return the score of an passed individual
        max_epochs: int(10)
            max number of iterations to search for the best individual
        force_mutate: bool(true)
            forces to mutate. if true and nothing changed it tries to mutate until something changed
        op: operation.lt
            compare operation to determine which score is better for a parent and an individual
            e. g. operator.gt means parent.score > child.score so the score of a child must be smaller to be
            the best of an epoch

        Returns
        -------

        """
        # todo: optional load/save best model

        if not callable(evaluator):
            raise TypeError("evaluator must be callable")

        parent = Individual(self.config)
        self.__evaluator_wrapper(evaluator, parent, 0)
        current_epoch = 0

        threads = [None] * self.num_children
        children = [None] * self.num_children

        while current_epoch < max_epochs:
            for i in range(self.num_children):
                # mutated = deepcopy(parent)
                mutated = pickle.loads(pickle.dumps(parent, -1))
                mutated.mutate(force_mutate)

                threads[i] = Thread(target=self.__evaluator_wrapper, args=(evaluator, mutated, i + 1))
                children[i] = mutated

            for t in threads:
                t.start()

            for t in threads:
                t.join()

            for child in children:
                if op(parent.score, child.score):
                    if verbose:
                        print("child %.2f has a better score than parent %.2f" % (child.score, parent.score))
                        plot_graph(parent, filename='tmp/mutation_%d.png' % current_epoch)
                    parent = child
            current_epoch += 1
