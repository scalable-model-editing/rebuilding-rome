import os
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

sys.path.append('/home/akshatgupta/KnowledgeEditing/model-editing')
from useful_functions import save_data

# Create a custom formatter function
def decimal_formatter(x, pos):
    return f'{x:.1e}'  # Adjust the number of decimal places as needed


if __name__ == '__main__':
    x_tick_size = 22
    y_tick_size = 22
    x_lim = 1000
    y_lim = 100
    axis_fontsize = 24
    legend_fontsize = 16


    metric_names = ['correct', 'f1', 'mcc', 'invalid']
    task_names = ['sst', 'mrpc', 'cola', 'rte']
    
    glue_eval = {'distance':{}}
    for task in task_names:
        glue_eval[task] = {}
        for metric in metric_names:
            glue_eval[task][metric] = {}


    algo = 'ROME'
    run = 'run_011'
    save_location = 'downstream_eval/' + algo + '_' + run + '/'
    os.makedirs(save_location, exist_ok=True)
    data_location = 'results/' + algo + '/' + run + '/glue_eval/'


    for filename in os.listdir(data_location):
        file_loc = data_location + filename
        if 'glue' in filename:
            with open(file_loc, "r") as f:
                data = json.load(f)

            if 'base' in filename:
                edit_num = 0
            else:
                edit_num = data['edit_num'] + 1
            #sample_num = data['sample_num']

            #plot distance data
            '''for layer in data['distance_from_original']:
                if layer not in glue_eval['distance']:
                    glue_eval['distance'][layer] = {}

                glue_eval['distance'][layer][edit_num] = data['distance_from_original'][layer]'''

            for task in task_names:
                if task in data:
                    for metric in metric_names:
                        glue_eval[task][metric][int(edit_num)] = data[task][metric]

    distance_data_location = 'results/' + algo + '/' + run + '/'
    for filename in os.listdir(distance_data_location):
        file_loc = distance_data_location + filename
        if '.json' in filename and filename != 'params.json':
            with open(file_loc, "r") as f:
                data = json.load(f)

            edit_num = data['case_id']
            #sample_num = data['sample_num']

            #plot distance data
            for layer in data['distance_from_original']:
                if layer not in glue_eval['distance']:
                    glue_eval['distance'][layer] = {}

                glue_eval['distance'][layer][edit_num] = data['distance_from_original'][layer]


    task_dict = {'sst':'SST2', 'mrpc':'MRPC', 'cola':'COLA', 'rte':'NLI'}
    run_title = {}
    task_colors = {'sst':'r', 'mrpc':'b', 'cola':'g', 'rte':'k'}
    #plot metrics individual with number of edits
    for metric in metric_names:
        plt.figure(figsize=(6.5, 5.5))
        for task in task_names:
            sorted_dict = sorted(glue_eval[task][metric].items(), key=lambda item: item[0])

            x, y = [], []
            count = 0
            for edit_num, correct in sorted_dict:
                x.append(count * 20)
                count += 1
                if metric in ['f1', 'accuracy']:
                    y.append(correct * 100)
                else:
                    y.append(correct / 200)

            plt.plot(x,y, label = task_dict[task], linewidth =3, color=task_colors[task])

        plt.legend(fontsize=legend_fontsize)
        plt.xlabel('Number of Edits', fontsize=axis_fontsize)
        if metric == 'correct':
            metric = 'accuracy'
        plt.ylabel(metric.upper(), fontsize=axis_fontsize)
        plt.xlim(0, x_lim)
        plt.ylim(0, y_lim)
        plt.tick_params(axis='x', labelsize=x_tick_size)
        plt.tick_params(axis='y', labelsize=y_tick_size)
        plt.tight_layout()

        if run in run_title:
            plt.savefig(save_location + algo + '_' + 'glue_' + metric + '_' + run_title[run] + '.png')
        else:
            plt.savefig(save_location + algo + '_' + 'glue_' + metric + '.png')
        plt.close()


    #plot distance as a function of number of edits
    metric = 'distance'
    x_store = []
    y_store = []
    for l, layer in enumerate(glue_eval[metric]):
        sorted_dict = sorted(glue_eval[metric][layer].items(), key=lambda item: item[0])

        x, y = [], []
        count = 0
        for edit_num, correct in sorted_dict:
            count += 1
            x.append(count)
            y.append(correct)

        x_store.append(x)
        y_store.append(y)

        if 'transformer' in layer:
            layer = layer.split('.')[2]

        if l == 0:
            plt.plot(x,y, linewidth =3, color = 'r', label = 'Layer ' + str(int(layer) + 1))
        else:
            plt.plot(x,y, linewidth =3, label = 'Layer ' + str(int(layer) + 1))
        
    plt.legend(fontsize=legend_fontsize)
    plt.xlabel('Number of Edits', fontsize=axis_fontsize)
    plt.ylabel('|Delta|', fontsize=axis_fontsize)
    plt.xlim(0, x_lim)
    plt.tick_params(axis='x', labelsize=x_tick_size)
    plt.tick_params(axis='y', labelsize=y_tick_size)
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(decimal_formatter))

    plt.tight_layout()


    if run in run_title:
        plt.savefig(save_location + algo + '_' + 'distance_' + run_title[run] + '.png')
    else:
        plt.savefig(save_location + algo + '_' + 'distance.png')
    plt.close()
            
    #print(len(y))
    #print(y)
    #save_data(algo + '_distance.pkl', [x_store,y_store])
    #plot glue performance as a function of number of edits