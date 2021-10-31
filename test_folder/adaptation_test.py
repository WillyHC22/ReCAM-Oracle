import os
import ast
import sys
sys.path.append(".")
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from oracle_subtask2.utils import get_summary_pseudogold, load_recam_dataset, load_tapt_dataset
from oracle_subtask2.create_gold_list_840 import create_list_gold_oracle_2


def recam_task_adaptation(model, tokenizer, task_number=1, dataset_name='xsum', split='train', gold_generator = 'tamamc', filename = '', dataset_path='save'):
    
    print('Load dataset')
    #recam_dataset = load_recam_dataset(task_number = 1, split = 'train')
    passages, summaries = load_tapt_dataset(dataset_name = dataset_name, split = 'train')
        
    print('Choose and mask gold candidates')
    if gold_generator == 'tamamc':
    #     gold_candidates = get_gold_candidates(passages, summaries, recam_dataset)
    #     golds = choose_gold(recam_dataset, gold_candidates)
    #     masked_summaries = get_gold_and_mask(summaries, golds, tokenizer)
        pass

    elif gold_generator == 'oracle':
        if os.path.isfile("oracle_subtask2/results/gold_answers_840.txt"):
            with open("oracle_subtask2/results/gold_answers_840.txt", mode="r", encoding="utf-8") as f:
                for line in f:
                    golds = ast.literal_eval(line)
        else:
            golds = create_list_gold_oracle_2(passages, summaries)[0]

        masked_summaries = get_gold_and_mask_oracle_2(summaries, golds, tokenizer)
 
#    elif gold_generator == 'sequence tagger':
#        ## TODO

    print('Get pseudo golds')
    pseudo_golds = get_summary_pseudogold(masked_summaries, golds, model, tokenizer, filename = '', dataset_path='save')
                                                                              
    return masked_summaries, golds, pseudo_golds


def get_gold_and_mask_oracle_2(summaries, golds, tokenizer):
    
    masked_summaries = []
    for i in tqdm(range(len(summaries))):
        gold = golds[i]
        if gold != "":
            gold = gold.lower()
            
            # corenlp can't receive % etc
            text = summaries[i].lower()

            # mask
            start_gold_index = text.index(gold)
            end_gold_index = start_gold_index + len(gold)
            text = text[:start_gold_index] + tokenizer.mask_token + text[end_gold_index:]
            
        else:
            text = None
        
        masked_summaries.append(text)
            
    return masked_summaries

if __name__ == "__main__":
    
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    masked_summaries, golds, pseudo_golds = \
        recam_task_adaptation(model, tokenizer, task_number=1, dataset_name='xsum', split='train', gold_generator = 'oracle', filename = '', dataset_path='save')
    
    with open("ReCAM_regen_XSUM_oracle2.txt", mode="w+", encoding="utf-8") as f:
        for i in range(len(masked_summaries)):
            f.write("{masked_summaries[i]}, {golds[i]}, {pseudo_golds[i]} \r\n")
