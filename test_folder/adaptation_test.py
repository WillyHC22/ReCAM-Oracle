import sys
sys.path.append(".")
from oracle_subtask2.utils import get_summary_pseudogold, load_recam_dataset, load_tapt_dataset



def recam_task_adaptation(model, tokenizer, task_number=1, dataset_name='xsum', split='train', gold_generator = 'tamamc', filename = '', dataset_path='save'):
    
    print('Load dataset')
    recam_dataset = load_recam_dataset(task_number = 1, split = 'train')
    passages, summaries = load_tapt_dataset(dataset_name = dataset_name, split = 'train')
        
    print('Choose and mask gold candidates')
    if gold_generator == 'tamamc':
    #     gold_candidates = get_gold_candidates(passages, summaries, recam_dataset)
    #     golds = choose_gold(recam_dataset, gold_candidates)
    #     masked_summaries = get_gold_and_mask(summaries, golds, tokenizer)
        pass
    elif gold_generator == 'oracle':
        golds = None
        masked_summaries = None
 
#    elif gold_generator == 'sequence tagger':
#        ## TODO

    print('Get pseudo golds')
    pseudo_golds = get_summary_pseudogold(masked_summaries, golds, model, tokenizer, filename = '', dataset_path='save')
                                                                              
    return masked_summaries, golds, pseudo_golds


