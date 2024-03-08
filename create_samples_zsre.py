#here we create 10 samples of size 500 to sample the counterfact dataset from to visualize model editing effects

from dsets.zsre import MENDQADataset
from transformers import AutoTokenizer
import random
import json

random.seed(37)

if __name__ == '__main__':

    write_flag = True
    n_samples = 5000
    output_filename = 'disabling_edits_zsre.json'


    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = MENDQADataset('data', tokenizer)


    #select indices with unique subjects in zsre
    all_subjects = {}
    for i in range(len(dataset)):
        subject = dataset.__getitem__(i)['requested_rewrite']['subject'].lower()
        
        if subject not in all_subjects:
            all_subjects[subject] = i
    
    #select n_sample indices from above unique subject examples
    all_indices = list(all_subjects.values())
    random.shuffle(all_indices)
    selected_indices = all_indices[:n_samples]

    #create hash dictionary for the selected indices
    selected_indices_dict = {}
    for i in range(len(dataset)):
        if i in selected_indices:
            selected_indices_dict[i] = True
        else:
            selected_indices_dict[i] = False

    print(sum(selected_indices_dict.values()))

    #save dictionary
    if write_flag:
        json_object = json.dumps(selected_indices_dict, indent=4)
        with open(output_filename , "w") as outfile:
            outfile.write(json_object)
