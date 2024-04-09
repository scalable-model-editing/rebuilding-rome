###This file measures the efficacy of making the current edit given a larger number of edits has been made in the past


import argparse
import json
import math
import os
import re

import matplotlib.pyplot as plt


def moving_average(data, window_size):
    average_array = []

    for i in range(len(data)):
        window = data[max(i - window_size + 1, 0) : i + 1]
        window_avg = sum(window) / len(window)
        average_array.append(window_avg)

    return average_array


def main(args):
    metrics = {
        "rewrite_prompts_probs": [],
        "paraphrase_prompts_probs": [],
        "neighborhood_prompts_probs": [],
    }

    algo = "ROME"
    run = args.run

    save_location = os.path.join("results", algo, run, "evals", "edit_scores")
    os.makedirs(save_location, exist_ok=True)
    data_location = os.path.join("results", algo, run)

    bucket_size = 5
    window_size = 10

    filename_regex = re.compile(r".*case_\d+.*\.json")

    sorted_filenames = sorted(
        filter(lambda x: filename_regex.match(x), os.listdir(data_location))
    )

    for filename in sorted_filenames:
        file_loc = os.path.join(data_location, filename)

        if not os.path.exists(file_loc):
            break

        with open(file_loc, "r") as f:
            data = json.load(f)

        for metric in metrics:
            if metric in ["rewrite_prompts_probs", "paraphrase_prompts_probs"]:
                success = [
                    element["target_new"] < element["target_true"]
                    for element in data["post"][metric]
                ]
            else:
                success = [
                    element["target_new"] > element["target_true"]
                    for element in data["post"][metric]
                ]

            value = sum(success) / len(success)
            metrics[metric].append(value)

    # making individual bar plots
    for metric in metrics:
        x, y = [], []
        for i in range(math.ceil(len(metrics[metric]) // bucket_size)):
            x.append(i)

            start_index = i * bucket_size
            end_index = min((i + 1) * bucket_size, len(metrics[metric]))
            y.append(sum(metrics[metric][start_index:end_index]))

        plt.bar(x, y)
        plt.savefig(os.path.join(save_location, "score_" + metric + ".png"))
        plt.close()

    metric_colors = {
        "rewrite_prompts_probs": "k",
        "paraphrase_prompts_probs": "b",
        "neighborhood_prompts_probs": "r",
    }
    metric_labels = {
        "rewrite_prompts_probs": "Efficacy Score",
        "paraphrase_prompts_probs": "Paraphrase Score",
        "neighborhood_prompts_probs": "Neighborhood Score",
    }

    # making overall plot
    plt.figure(figsize=(6.5, 6))
    for metric in metrics:
        x, y = [], []
        for i in range(math.ceil(len(metrics[metric]) // bucket_size)):
            x.append(i * bucket_size)

            start_index = i * bucket_size
            end_index = min((i + 1) * bucket_size, len(metrics[metric]))

            y.append((sum(metrics[metric][start_index:end_index]) / bucket_size) * 100)

        y_avg = moving_average(y, window_size)

        plt.plot(x, y, linestyle="--", color=metric_colors[metric], linewidth=0.2)
        plt.plot(
            x,
            y_avg,
            color=metric_colors[metric],
            label=metric_labels[metric],
            linewidth=2,
        )

    plt.legend(loc="upper left", bbox_to_anchor=(0.45, 1.28), ncol=1, fontsize=14)
    plt.xlabel("Number of Edits", fontsize=20)
    plt.ylabel("Edit Accuracy", fontsize=20)
    plt.xlim(0, len(sorted_filenames))
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_location, "editing_score.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, help="Name of run to evaluate")
    args = parser.parse_args()
    main(args)
