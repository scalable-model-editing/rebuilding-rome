import os
import sys
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

sys.path.append('/home/akshatgupta/KnowledgeEditing_local/disabling-edits')
from useful_functions import save_data


def get_distribution(input_list):
    frequency = {}

    for element in input_list:
        if element not in frequency:
            frequency[element] = 1
        else:
            frequency[element] += 1

    #convert frequency to distribution
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
    for sentence in data['post']['text']:
        words.extend(sentence.lower().replace('.', '').split(' '))

    distribution = get_distribution(words)
    entropy = get_entropy(distribution)

    normalized_entropy = entropy / math.log2(len(distribution))

    return entropy, normalized_entropy



if __name__ == '__main__':
    x_tick_size = 22
    y_tick_size = 22
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 24
    legend_fontsize = 16

    with open('data/disabling_edits_counterfact.json') as json_file:
        selected_indices  = json.load(json_file)

    algo = 'ROME'
    run = 'run_000'
    save_location = 'downstream_eval/plots/' + algo + '/' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = '/home/akshatgupta/KnowledgeEditing/model-editing/results/' + algo + '/' + run + '/'


    distance_files = {}
    entropy_files = {}


    normalized_entropy = []
    entropy = []
    distances = {}
    for filename in os.listdir(data_location):
        file_loc = data_location + filename
        if 'case' in filename:
            with open(file_loc, "r") as f:
                data = json.load(f)

            if not selected_indices[str(data['case_id'])]:
                continue

            #collect variables
            sample_entropy, sample_normalized_entropy = get_generation_entropy(data)

            normalized_entropy.append(sample_normalized_entropy)
            entropy.append(sample_entropy)
            entropy_files[filename] = sample_normalized_entropy

            for layer in data['distance']:
                if layer not in distances:
                    distances[layer] = []
                    distance_files[layer] = {}

                distances[layer].append(data['distance'][layer])
                distance_files[layer][filename] = data['distance'][layer]

    for layer in distances:
        plt.scatter(distances[layer], normalized_entropy, color = 'r')

        plt.xlabel('Normalized Distance', fontsize=axis_fontsize)
        plt.ylabel('Normalized Entropy', fontsize=axis_fontsize)
        plt.tick_params(axis='x', labelsize=x_tick_size)
        plt.tick_params(axis='y', labelsize=y_tick_size)
        plt.tight_layout()

        # Format the x-axis ticks
        ax = plt.gca()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        ax.xaxis.set_major_formatter(formatter)

        plt.savefig(save_location + 'disabling_normalized_entropy_layer_{}.png'.format(layer))
        plt.close()

    for layer in distances:
        plt.scatter(distances[layer], entropy, color = 'r')

        plt.xlabel('Normalized Distance', fontsize=axis_fontsize)
        plt.ylabel('Entropy', fontsize=axis_fontsize)
        plt.tick_params(axis='x', labelsize=x_tick_size)
        plt.tick_params(axis='y', labelsize=y_tick_size)
        plt.tight_layout()

        # Format the x-axis ticks
        ax = plt.gca()
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1,1))
        ax.xaxis.set_major_formatter(formatter)

        plt.savefig(save_location + 'disabling_entropy_layer_{}.png'.format(layer))
        plt.close()
                

    sorted_entropy = sorted(entropy_files.items(), key=lambda item: item[1])

    print('PRINTING ENTROPY')
    for filename, entropy in sorted_entropy[:10]:
        print(filename, entropy)
    print()

    for layer in distance_files:
        distance_f = sorted(distance_files[layer].items(), key=lambda item: item[1], reverse=True)
        
        print('PRINTING ENTROPY Layer', layer)
        for filename, distance in distance_f[:10]:
            print(filename, distance)
        print()

    print('TOTAL DONE:', len(sorted_entropy))
