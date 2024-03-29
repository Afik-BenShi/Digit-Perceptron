"""Neural network things"""
# %%
# Was built while reading Michael Nielsens awesome book
# which is available for free at http://neuralnetworksanddeeplearning.com/
# and while watching the great 3 blue 1 brown video series by Grant Sanderson,
# which is also avilable for free at
# https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
#
import json
from glob import glob
from typing import Any

import numpy as np

import mnist_loader as mnl


def sigmoid(x):
    """just a regular sigmoid working with numpy"""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(x):
    """sigmoid derivative working with numpy"""
    return sigmoid(x) * (1 - sigmoid(x))


class Network:
    """
    Neural Network class
    --------------------

    Includes all layers and functions for the network to work properly


    Funtions
    --------
        netword_eval(self, input_image):
            evaluates the network with the input image
    """

    def __init__(
        self,
        sizes: "list[int]",
        output_lbls: "list[Any]" = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    ) -> "Network":
        """

        Parameters
        ----------
            layer_sizes : array like - size of each layer in the network
            output_lbls : array like - labels of the output in order
        """
        self.__output_lbls = list(output_lbls)

        # 1st layer is img, last is lbl
        sizes = [784] + sizes + [len(output_lbls)]
        self.__layer_num = len(sizes)

        self.__biases = [np.random.randn(n, 1) for n in sizes[1:]]

        self.__weights = [
            np.random.randn(*shape) for shape in zip(sizes[1:], sizes[:-1])
        ]

    # <Overriden functions>
    def __repr__(self):
        txt = "Neural network \n"
        for layer in self.__weights:
            txt += f"{len(layer)} -> "
        txt += f"{self.__output_lbls}"
        return txt

    def __len__(self):
        return self.__layer_num

    def __copy__(self):
        net = Network([1], self.__output_lbls.copy())
        net._set_weights_and_biases(
            list(self.__weights).copy(), list(self.__biases).copy()
        )
        return net

    def _set_weights_and_biases(self, weights, biases) -> None:
        """setting the matricies values"""
        self.__weights = weights
        self.__biases = biases
        self.__layer_num = len([b.shape[0] for b in biases])

    # </Overriden functions>

    # <Forward and backward ealuations>
    def evaluate(self, input_image: "mnl.LabeledImage") -> "np.ndarray[int]":
        """Evaluates the network with the input image

        Returns
        -------
        output activation layer
        """
        prev_layer = input_image.image_1D  # assign input to last layer
        for b, w in zip(self.__biases, self.__weights):
            prev_layer = sigmoid(np.dot(w, prev_layer) + b)
        return prev_layer

    def backprop(self, input_image: "mnl.LabeledImage") -> "tuple[list[float]]":
        """Evaluates input image and compares output to answer.
        propegates the error and calculates gradiants to be used in GD algoritm

        Returns
        --------
        (grad_weights, grad_biases)"""
        nabla_b = [np.zeros(b.shape) for b in self.__biases]
        nabla_w = [np.zeros(w.shape) for w in self.__weights]
        # feedforward
        act = input_image.image_1D
        # list to store all the activations, layer by layer
        activations = [act]
        zees = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.__biases, self.__weights):
            z = np.dot(w, act) + b
            zees.append(z)
            act = sigmoid(z)
            activations.append(act)
        # backward pass
        delta = self.cost_deriv(activations[-1], input_image.label) * sigmoid_deriv(
            zees[-1]
        )
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.__layer_num):
            z = zees[-i]
            w = self.__weights[-i + 1]
            delta = np.dot(w.transpose(), delta) * sigmoid_deriv(z)
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        return (nabla_b, nabla_w)

    def batch_evaluate(self, test_data: "list[mnl.LabeledImage]") -> int:
        """Run the neural network over a labled test data
        and return how many guesses were successful"""
        lbls = self.__output_lbls
        test_res = [
            (lbls[np.argmax(self.evaluate(img).flatten())], img) for img in test_data
        ]
        return sum(int(guess == img) for (guess, img) in test_res)

    def detect(self, input_image: "mnl.LabeledImage", detailed: bool = False) -> "Any":
        """try to guess what number is written in the input image"""
        activ = self.evaluate(input_image)
        lbls = self.__output_lbls
        detection = lbls[np.argmax(activ)]
        if detailed:
            input_image.show()
            confidance = np.max(activ) * 100
            print(f"Network guessed {detection} with {confidance:.1f}% confidance")
            return detection
        return detection

    def cost_deriv(
        self, net_res: "np.ndarray[float]", answer: "Any"
    ) -> "np.ndarray[float]":
        """
        Calculates the derivative of a quadratic cost function.

        Parameters
        ----------
            net_res : array - The output actication layer from evaluation
            answer  : The label of the real answer as it is saved in the networks labels
        """
        assert answer in self.__output_lbls
        # construct wanted activation vector from answer
        _n = len(self.__output_lbls)
        ans_vec = np.zeros((_n, 1))
        ans_vec[self.__output_lbls.index(answer)] = 1
        # calculate and return the derivative
        return net_res - ans_vec

    # </Forward and backward ealuations>

    # <Stochastic gradient descent>
    def sgd(
        self,
        training_data: "list[mnl.LabeledImage]",
        epochs: int,
        batch_size: int,
        learning_rate: float,
        test_data: "list[mnl.LabeledImage]" = None,
        progress: str = "all",
    ):
        """
        Train the network using the training data.
        Data is divided into small batches,
        gradient is decided for each batch using back propegation,
        and a step is taken in the gradient direction.

        Progress bar system is implemented, controlled by the progress variable

        Parameters
        ----------
            training_data : list - labled images to train the data from
            epochs        : int - the number of times to train over the training data
            batch_size    : int - the size of each batch of data to create a gradient
            learning_rate : float - step size taken in the direction of the gradient
            test_data     : list - labled images to test the network on after each ttraining
            progress      : str - controls the progress bar.
                                  "no"      - no progress bar
                                  "epochs"  - progress bar for each epoch
                                  "all"     - progress bar for the all process
        """
        if test_data and progress != "no":  # print test before training
            n_test = len(test_data)
            print("Testing network with before training")
            print("testing data ", end="")
            print(f"\rBenchmark result: {self.batch_evaluate(test_data)}/{n_test}")

        bar_len = 25
        n = len(training_data)
        for i in range(epochs):
            np.random.shuffle(training_data)
            batches = [
                training_data[j : j + batch_size] for j in range(0, n, batch_size)
            ]

            if progress == "epochs":
                print(f'Epoch {i+1} of {epochs}. Progress |{" " * bar_len}|', end="")
            elif progress == "all":
                prog = (i * bar_len) // epochs
                print(
                    f'\rTraining progress |{"#" * prog + " " * (bar_len-prog)}|', end=""
                )

            batch_cnt = n // batch_size
            for j, batch in enumerate(batches):
                self.update_batch(batch, learning_rate)  # train over batch

                last = 0  # progress bar
                prog = int(j / batch_cnt * bar_len)
                if prog > last and progress == "epochs":
                    last = prog
                    print(
                        f'\rEpoch {i+1} of {epochs}.                              \
                           Progress |{"#" * prog + " " * (bar_len-prog)}|',
                        end="",
                    )
            if progress == "epochs":
                print(f'\rEpoch {i+1} of {epochs}. Progress |{"#" * bar_len}|')

            if test_data and progress == "epochs":
                n_test = len(test_data)
                print("testing data", end="")
                print(
                    f"\rBenchmark result: {self.batch_evaluate(test_data)}/{n_test}\n"
                )

        if progress == "all":
            print(f'\rTraining progress |{"#" * bar_len}|')
        if test_data and progress != "epochs":
            n_test = len(test_data)
            print("\nTest after training")
            print("testing data", end="")
            print(f"\rBenchmark result: {self.batch_evaluate(test_data)}/{n_test}\n")

    def update_batch(
        self, batch: "list[mnl.LabeledImage]", learning_rate: float
    ) -> None:
        """Update network based on a batch"""
        nabla_b = [np.zeros(b.shape) for b in self.__biases]
        nabla_w = [np.zeros(w.shape) for w in self.__weights]
        for image in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(image)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update weights and biases
        self.__weights = [
            w - (learning_rate / len(batch)) * nw
            for w, nw in zip(self.__weights, nabla_w)
        ]
        self.__biases = [
            b - (learning_rate / len(batch)) * nb
            for b, nb in zip(self.__biases, nabla_b)
        ]

    # </Stochastic gradient descent>

    # <json stuff>
    def save_to_json(self) -> str:
        """turns the data to json format and saves it.
        Returns the file name of the saved network.
        """

        json_dict = {
            f"layer{i}": {  # number layers in json file
                "weights": list(self.__weights[i].flatten()),
                "w_shape": list(self.__weights[i].shape),
                "biases": list(self.__biases[i].flatten()),
            }
            for i in range(len(self.__biases))
        }

        json_dict["length"] = len(self)
        json_dict["labels"] = self.__output_lbls

        json_str = json.dumps(json_dict, indent=4)  # turn into json format

        name = ""
        try:
            names = glob("saves\\*.json")
            _id = max((int("".join(i for i in name if i.isdigit())) for name in names))
            name = f"saves\\save_{_id+1}.json"
        except ValueError:
            name = "saves\\save_0.json"
        finally:
            with open(name, "w", encoding="utf-8") as file:
                file.write(json_str)
        return name

    @staticmethod
    def load_from_json(path: str = None) -> "Network":
        """creates a layer according to saved json file
        if path is None, takes the last one"""

        saves = glob("saves\\*.json")
        if path is None:
            # choose most recent save
            _id = max((int("".join(i for i in save if i.isdigit())) for save in saves))
            path = f"saves\\save_{_id}.json"
        if path not in saves:
            # assert savefile exists
            return None

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        n = data["length"]
        weights = []
        biases = []
        for i in range(n - 1):
            l_dict = data[f"layer{i}"]

            w = np.array(l_dict["weights"])
            w = w.reshape(*l_dict["w_shape"])
            weights.append(w)

            b = np.array(l_dict["biases"])
            b = b.reshape((len(b), 1))
            biases.append(b)

        return Network.__from_data(weights, biases, data["labels"])

    # </json stuff>

    @staticmethod
    def __from_data(
        weights: "np.ndarray[float]", biases: "np.ndarray[float]", labels: "list[Any]"
    ) -> "Network":
        """Creates a network from the given parameters.
        Parameters
        ----------
            weights : array like - the weights for each of the connections between neurons
            biases  : array like - the biases for each of the neurons in the network
            labels  : iterable - the labels of the output layer
        """

        class NetworkFactory(Network):
            """Factory to create networks from given matricies values"""

            def __init__(self, weights, biases, lbls):
                super().__init__([1], lbls)
                super()._set_weights_and_biases(weights, biases)

            def to_net(self):
                """copy to Network instance"""
                return Network.__copy__(self)

        factory = NetworkFactory(weights, biases, list(labels).copy())
        return factory.to_net()


# %%
