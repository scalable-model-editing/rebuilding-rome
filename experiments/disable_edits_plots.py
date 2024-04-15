import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, ScalarFormatter

sys.path.append("/workspace/rebuilding-rome")
from useful_functions import save_data


def get_distribution(input_list):
    frequency = {}

    for element in input_list:
        if element not in frequency:
            frequency[element] = 1
        else:
            frequency[element] += 1

    # convert frequency to distribution
    for element in frequency:
        frequency[element] /= len(input_list)

    return frequency


def get_entropy(input_dict):
    entropy = 0
    probabilities = input_dict.values()

    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def get_generation_entropy(data):
    words = []

    # get list of generations
    generations = []
    for key in data["post"].keys():
        if "probs" not in key:
            continue
        for item in data["post"][key]:
            generations.append(item["generated_text"])

    for sentence in generations:
        words.extend(sentence.lower().replace(".", "").split(" "))

    distribution = get_distribution(words)
    entropy = get_entropy(distribution)

    normalized_entropy = entropy / math.log2(len(distribution))

    return entropy, normalized_entropy


# Create a custom formatter function
def decimal_formatter(x, pos):
    return f"{x:.1e}"  # Adjust the number of decimal places as needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        help="run dir name",
    )
    args = parser.parse_args()

    x_tick_size = 16
    y_tick_size = 16
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 24
    legend_fontsize = 16

    algo = "ROME"
    run = args.run
    save_location = os.path.join("results", algo, run, "evals")
    os.makedirs(save_location, exist_ok=True)
    data_location = os.path.join("results", algo, run)

    distance_files = {}
    entropy_files = {}

    normalized_entropy = []
    entropy = []
    distances = {}
    for filename in os.listdir(data_location):
        file_loc = os.path.join(data_location, filename)
        if "case" in filename:
            with open(file_loc, "r") as f:
                data = json.load(f)

            # collect variables
            sample_entropy, sample_normalized_entropy = get_generation_entropy(data)

            normalized_entropy.append(sample_normalized_entropy)
            entropy.append(sample_entropy)
            entropy_files[filename] = sample_normalized_entropy

            for layer in data["distance_from_original"]:
                if layer not in distances:
                    distances[layer] = []
                    distance_files[layer] = {}

                distances[layer].append(data["distance_from_original"][layer])
                distance_files[layer][filename] = data["distance_from_original"][layer]

    for layer in distances:
        # plt.scatter(distances[layer], normalized_entropy, color = 'r')
        data = pd.DataFrame(
            {
                "Normalized Distance": distances[layer],
                "Normalized Entropy": normalized_entropy,
            }
        )
        sns.scatterplot(
            x="Normalized Distance", y="Normalized Entropy", data=data, s=200
        )

        plt.xlabel("|Delta|", fontsize=axis_fontsize)
        plt.ylabel("Normalized Entropy", fontsize=axis_fontsize)
        plt.tick_params(axis="x", labelsize=x_tick_size)
        plt.tick_params(axis="y", labelsize=y_tick_size)
        plt.ylim(0, 1)
        plt.tight_layout()

        # Format the x-axis ticks
        ax = plt.gca()
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True)
        # formatter.set_powerlimits((-1,1))
        # ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(FuncFormatter(decimal_formatter))

        plt.savefig(
            os.path.join(
                save_location, "disabling_normalized_entropy_layer_{}.png".format(layer)
            )
        )
        plt.close()

    for layer in distances:
        plt.scatter(distances[layer], entropy, color="r")

        plt.xlabel("Normalized Distance", fontsize=axis_fontsize)
        plt.ylabel("Entropy", fontsize=axis_fontsize)
        plt.tick_params(axis="x", labelsize=x_tick_size)
        plt.tick_params(axis="y", labelsize=y_tick_size)
        plt.tight_layout()

        # Format the x-axis ticks
        ax = plt.gca()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.xaxis.set_major_formatter(formatter)

        plt.savefig(
            os.path.join(save_location, "disabling_entropy_layer_{}.png".format(layer))
        )
        plt.close()

    sorted_entropy = sorted(entropy_files.items(), key=lambda item: item[1])

    print("PRINTING ENTROPY")
    for filename, entropy in sorted_entropy[:10]:
        print(filename, entropy)
    print()

    for layer in distance_files:
        distance_f = sorted(
            distance_files[layer].items(), key=lambda item: item[1], reverse=True
        )

        print("PRINTING ENTROPY Layer", layer)
        for filename, distance in distance_f[:10]:
            print(filename, distance)
        print()

    print("TOTAL DONE:", len(sorted_entropy))
