"""Script that trains and evaluates a single neuron."""

import numpy as np

from dataset import read_datasets
from plotting import plot_results
import neuron


def train_neuron(
    data_path: str,
    linearly_separable: bool,
    batch_size: int,
    learning_rate: float,
    n_epochs: int,
):
    """Load datasets and train and test the neural network.

    Args:
        data_path: path where the .mat files are stored
        linearly_separable: if True, linear separable dataset is loaded
                            if False, linear non-separable dataset is loaded
        batch_size: number of examples in a mini-batch
        learning_rate: the learning rate to use in gradient descend
        n_epochs: the number of epochs to train the neuron for
    """

    # Load datasets
    datasets = read_datasets(data_path, linearly_separable)
    n_inputs = datasets.train.n_inputs
    n_outputs = datasets.train.n_outputs
    assert n_inputs == 2
    assert n_outputs == 1

    # Initialize weights and bias
    w = np.random.normal(0, 0.58, (n_inputs))
    b = np.zeros(n_outputs)

    # Train neuron
    epoch = 0
    while epoch < n_epochs:
        # Get training data of next mini-batch
        batch_inputs, batch_labels = datasets.train.get_next_batch(batch_size)

        # Update weights and bias
        w, b = neuron.update_weights(batch_inputs, batch_labels, w, b, learning_rate)

        if datasets.train.is_epoch_completed(batch_size):
            # Epoch completed: calculate current loss of validation set
            epoch += 1
            loss = neuron.loss_function(
                datasets.validation.inputs, datasets.validation.labels, w, b
            )
            print(f"epoch: {epoch:4d}   validation loss: {loss:.6f}")

    # Evaluate trained neuron
    _, train_accuracy = neuron.evaluate_prediction(
        datasets.train.inputs, datasets.train.labels, w, b
    )
    _, validation_accuracy = neuron.evaluate_prediction(
        datasets.validation.inputs, datasets.validation.labels, w, b
    )
    predicted_test_labels, test_accuracy = neuron.evaluate_prediction(
        datasets.test.inputs, datasets.test.labels, w, b
    )

    print()
    print(f"Training accuracy: {100*train_accuracy:.2f}%")
    print(f"Validation accuracy: {100*validation_accuracy:.2f}%")
    print(f"Test accuracy: {100*test_accuracy:.2f}%")

    plot_results(datasets.test.inputs, datasets.test.labels, predicted_test_labels)


if __name__ == "__main__":

    train_neuron(
        data_path="./data",
        linearly_separable=False,
        batch_size=100,
        learning_rate=100,
        n_epochs=100,
    )
