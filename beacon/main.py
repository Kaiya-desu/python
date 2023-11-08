import argparse
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt


# ******************************************************************************
# NETWORK IMPLEMENTATION
# ******************************************************************************
def calculate_mse(result, expected):
    return np.sum((result - expected)**2)/len(result)


def unipolar_activation(u):
    warnings.filterwarnings('ignore')
    return 1 / (1 + np.exp(-u))


def unipolar_derivative(u):
    a = unipolar_activation(u)
    return a * (1.0 - a)


def build_network(input_size, hidden_size, output_size):
    network = []
    network.append(np.random.rand(hidden_size, input_size + 1) * 2 - 1)
    network.append(np.random.rand(output_size, hidden_size + 1) * 2 - 1)
    return network


def activate(network, observation):
    responses = []
    layer_input = observation
    for layer in network:
        layer_input = np.append(layer_input, np.ones([1, layer_input.shape[1]]), 0)
        response = unipolar_activation(layer @ layer_input)
        responses.append(response)
        layer_input = response
    classification = responses[-1]
    return classification, responses


def calculate_error_gradients(network, responses, expected_classification):
    gradients = []
    error = responses[-1] - expected_classification
    for layer, response in zip(reversed(network), reversed(responses)):
        gradient = error + unipolar_derivative(response)
        gradients.append(gradient)
        error = layer.T @ gradient
        error = error[1:, :]
    return reversed(gradients)


def calculate_weights_changes(network, observation, responses, gradients, learning_factor):
    layer_inputs = [observation] + responses[:-1]
    weights_changes = []
    for layer, layer_input, gradient in zip(network, layer_inputs, gradients):
        layer_input = np.append(layer_input, np.ones([1, layer_input.shape[1]]), 0)
        change = layer_input @ gradient.T * learning_factor
        weights_changes.append(change.T)
    return weights_changes


def adjust_weights(network, weights_changes):
    new_network = []
    for layer, adjustment in zip(network, weights_changes):
        new_layer = layer - adjustment
        new_network.append(new_layer)
    return new_network


def fit(network, observation, expected_classifications, learning_factor, epochs):
    mse_values = []
    for epoch in range(epochs):
        classifications, responses = activate(network, observation)
        print(f'For epoch {epoch} value of MSE is {calculate_mse(classifications, expected_classifications)}')
        mse = calculate_mse(classifications, expected_classifications)
        mse_values.append(mse)
        gradients = calculate_error_gradients(network, responses, expected_classifications)
        weight_changes = calculate_weights_changes(network, observation, responses, gradients, learning_factor)
        network = adjust_weights(network, weight_changes)
    return mse_values, classifications


# ******************************************************************************
# HELPER FUNCTIONS
# ******************************************************************************
def read_data_from_csv(csv_file_location):
    data = pd.read_csv(csv_file_location)

    observations = data.iloc[:, :-2].to_numpy().T
    classifications = data.iloc[:, [3, 4]].to_numpy()
    categories = np.unique(classifications, axis=0)

    category_encoding = {}
    for code, name in enumerate(categories):
        hash_name = str(name[0]) + ' ' + str(name[1])
        category_encoding[hash_name] = code

    classifications_to_category = []
    for name in classifications:
        classifications_to_category.append(category_encoding[str(name[0]) + ' ' + str(name[1])])

    one_hot_encoder = np.eye(len(categories))
    classifications = one_hot_encoder[classifications_to_category].T

    category_decoder = {v: k for k, v in category_encoding.items()}

    return observations, classifications, category_decoder


def generate_csv_file(classifications):
    x = []
    y = []
    for classification in classifications:
        split_classification = classification.split()
        x.append(split_classification[0])
        y.append(split_classification[1])

    data_to_csv = {
        'x': x,
        'y': y
    }
    df = pd.DataFrame(data_to_csv)
    df.to_csv('output.csv')


def convert_classifications(classifications, category_decoder):
    classifications = classifications.argmax(axis=0)
    readable_classifications = []
    for classification in classifications:
        readable_classifications.append(category_decoder[classification])
    return readable_classifications


def positive_int(str_val):
    val = int(str_val)
    if not val > 0:
        raise ValueError(f'value must be greater than 0')
    return val


def positive_float(str_val):
    val = float(str_val)
    if not val > 0:
        raise ValueError(f'value must be greater than 0')
    return val


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='CSV file with data, last column must be as expected response')
    parser.add_argument('-s', '--hidden_size', type=positive_int, required=True,
                        help='Size of hidden layer')
    parser.add_argument('-l', '--learning_factor', type=positive_float, required=True,
                        help='Epsilon, learning factor')
    parser.add_argument('-e', '--epochs', type=positive_int, required=True,
                        help='Number of epochs')
    return parser.parse_args()


# drukowanie
def print_classifications(classifications, category_decoder):
    classifications = classifications.argmax(axis=0)
    for obs_id, classification in enumerate(classifications):
        print(f'{obs_id} -> {category_decoder[classification]}')


def mse_diagram(mse_values):
    plt.plot(range(1, len(mse_values)+1), mse_values)
    plt.ylabel("MSE values")
    plt.xlabel("EPOCHS")
    plt.show()


# ******************************************************************************
# MAIN
# ******************************************************************************
def main():
    args = parse_arguments()
    observation, classification, category_decoder = read_data_from_csv(args.input_file)

    network = build_network(observation.shape[0], args.hidden_size, classification.shape[0])
    # blind_classification, _ = activate(network, observation)
    # print_classifications(classification, category_decoder)
    mse_values, classifications = fit(network, observation, classification, args.learning_factor, args.epochs)
    mse_diagram(mse_values)
    classifications = convert_classifications(classifications, category_decoder)
    generate_csv_file(classifications)


if __name__ == '__main__':
    main()
