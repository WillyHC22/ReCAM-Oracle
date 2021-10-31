from pywsd.lesk import adapted_lesk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity


# Build the three filters used for finding non specific concept and for MCQ

#Build the Lemma filter
def lemmatize_sentence(token_sentence):
    """Take a tokenized sentence and return the lemmatized tokenized version of the sentence"""
    wnl = WordNetLemmatizer()
    lemm_sentence = []
    for word, tag in pos_tag(token_sentence):
        if tag.startswith("NN"):
            lemm_sentence.append(wnl.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            lemm_sentence.append(wnl.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            lemm_sentence.append(wnl.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            lemm_sentence.append(wnl.lemmatize(word, pos='r'))
        else:
            lemm_sentence.append(word)
    return lemm_sentence

def filter_lemma(passage, words):
    """Lemmatize all the words in a tokenized passage and a list of words, then deletes the words 
    which lemmas appear in the passage
    passage [list of str]
    words [list of str]: list of words gotten after first filtering through non specificity score'
    return a list of answers after lemma filter"""
    lem_passage = lemmatize_sentence(passage)
    lem_words = lemmatize_sentence(words)
    counter = 0
    answers = []
    for word in lem_words:
        if word not in lem_passage:
            answers.append(words[counter])
        counter += 1
    return answers



#Build the Synonyms and Antonyms filter
def generate_list_syn_ant(word):
    """Returns a single list of every synonyms and antonyms for a given world using wordnet
    word [str]"""
    syn_ant = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.name() not in syn_ant:
                syn_ant.append(l.name())
            if l.antonyms():
                if l.antonyms()[0].name() not in syn_ant:
                    syn_ant.append(l.antonyms()[0].name())
    return syn_ant

def filter_syn_ant(passage, words):
    """Create a pool of synonyms/antonyms of every noun/verb in the passage, then filter the
    noun/vern of the passage based of their appartenance in this list
    word [str]: word in the sentence that you want to apply the filter on
    return list of filtered answer
    """
    answers = [word for word in words if not (set(generate_list_syn_ant(word)) & set(passage))]
    return answers



#Build the Similarity filter
def filter_similarity(embedding_dict, passage, words):
    answers = []
    for word in words:
        #Check if the potential gold answer exists in the glove embedding
        if word.lower() in embedding_dict:
            word_emb = embedding_dict[word.lower()]
            similarity_check = True
            for word_p in passage:
                #Check if the word in the passage exists in the glove embedding
                if word_p.lower() in embedding_dict:
                    word_p_emb = embedding_dict[word_p.lower()]
                    cos_sim = cosine_similarity(word_emb.reshape(1,-1), word_p_emb.reshape(1,-1))
                    if cos_sim[0][0] > 0.85:
                        similarity_check = False
                        break
                else:
                    pass
            if similarity_check:
                answers.append(word)
        else:
            pass
    return answers




###

import re
import ast
from tqdm import tqdm
import torch
import random
import datasets
import os
import gzip
import json
import pandas as pd
from datetime import date

def get_summary_pseudogold(masked_summaries, golds, model, tokenizer, filename = '', dataset_path='save', top_selection=10):
    
    pattern = re.compile("[A-Za-z]+")

    pg_tokens_list = []
    for i in tqdm(range(len(masked_summaries))):
        gold = golds[i]
        if gold != "":

            # tokenize, convert to ids then tensor
            tokens = tokenizer.tokenize(masked_summaries[i])
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            token_tensor = torch.LongTensor([token_ids])

            # provide position ids explicitly
            position_ids = list(range(0, token_tensor.size(1)))
            position_id_tensor = torch.LongTensor([position_ids])

            masked_idx = [i for i, token in enumerate(tokens) if tokenizer.mask_token in token][0]
            # get the top 10 predictions of the masked token
            with torch.no_grad():
                outputs = model(input_ids=token_tensor, position_ids=position_id_tensor)
                predictions = outputs[0][0, masked_idx].topk(top_selection)

            # tokenizer is different than TA-MAMC, trying to remove word pieces
            pg_tokens = []
            for index in predictions.indices:
                token = tokenizer.convert_ids_to_tokens([index])[0]
                if gold not in token and token not in gold:
                    token = token[1:] if token[0] == 'Ġ' else token
                    if len(token) > 2 and pattern.fullmatch(token):
                        pg_tokens.append(token.lower())

            pg_tokens = list(set(pg_tokens))
            random.shuffle(pg_tokens)
            chosen_pg_tokens = pg_tokens[:4]
        else:
            chosen_pg_tokens = None 
        
        pg_tokens_list.append(chosen_pg_tokens)
        
    df_result = pd.DataFrame({'texts':masked_summaries, 'golds':golds, 'pseudo_golds':pg_tokens_list})
    
    df_result_filtered = df_result[~df_result['text'].isna()]
    df_result_filtered = df_result_filtered[df_result_filtered['pseudo_golds'] != '[]']
    df_result_filtered['len_pseudo_golds'] = df_result_filtered.apply(lambda x: len(ast.literal_eval(x['pseudo_golds'])), axis=1)
    df_result_filtered = df_result_filtered[df_result_filtered['len_pseudo_golds']>=4]
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        
    if filename == '':
        filename = date.today().strftime("%d%m%Y")
        
    df_result[['texts', 'golds', 'pseudo_golds']]\
        .to_csv(dataset_path + '/' + filename + '_masked_summary_golds_pseudo_options_filtered.csv', index=False)
    
    df_result_filtered[['texts', 'golds', 'pseudo_golds']]\
        .to_csv(dataset_path + '/' + filename + '_masked_summary_golds_pseudo_options_filtered.csv', index=False)
            
    return pg_tokens_list


def load_recam_dataset(task_number = 1, split = 'train'):
    
    with open(os.getcwd()[:-17] + 'data/recam/Task_' + str(task_number) + '_' + split + '.jsonl', 'r') as json_file:
        json_list = list(json_file)
    recam_dataset = [json.loads(json_str) for json_str in json_list]
    
    return recam_dataset


def load_tapt_dataset(dataset_name = 'xsum', split = 'train'):

    if dataset_name == 'xsum':

        passage = datasets.load_dataset('xsum')[split]['document']
        summary = datasets.load_dataset('xsum')[split]['summary']

    elif dataset_name == 'newsroom':

        path = os.getcwd()[:-17] + 'data/newsroom/release/' + split + '.jsonl.gz'
        dataset = []
        with gzip.open(path) as f:
            for ln in f:
                obj = json.loads(ln)
                dataset.append(obj)

        passage = [dataset[i]['text'] for i in range(len(dataset))]
        summary = [dataset[i]['summary'] for i in range(len(dataset))]

    elif dataset_name == 'cnn':

        passage = datasets.load_dataset('cnn_dailymail', '3.0.0')[split]['article']
        summary = datasets.load_dataset('cnn_dailymail', '3.0.0')[split]['highlights']
        
    return passage, summary