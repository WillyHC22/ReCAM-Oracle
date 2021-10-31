import numpy as np
from pywsd.lesk import adapted_lesk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from datasets import load_dataset 

from oracle_subtask2.utils import filter_lemma, filter_syn_ant, filter_similarity


lemmatizer = WordNetLemmatizer()

def NonSpecificity_Scorer(passage, summary, word):
    """
    Use Adapted Lesk Algorithm to give sense to the word, then get his synset and return his 
    distance to the root which represents the score
    passage [str] : passage to give sense to the word
    summary [str] : summary of the given passage
    word [str] : word that will be converted to score for non specificity 
    return a score [int] between 0 and 17 (0 being the root, lower means more abstract)
    """

    #The context used, according to the paper, is the concatenated passage with its summary
    context = passage + summary
    synset = adapted_lesk(context, word)
    #We automatically exclude the words for which the synset returns no type (e.g. ex-Italy)
    if synset is None:
        score = 17
    else:
        score = min([len(path) for path in synset.hypernym_paths()])
    return score



def create_embedding_dict_6(embedding_txt="glove.6B.50d.txt"):
    """embedding_txt [str] is the path to a txt file containing word vector association in 
    the format of GloVe."""
    embeddings_dict = {}
    with open(embedding_txt, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def create_embedding_dict_840(embedding_txt="glove.840B.300d.txt"):
    embeddings_dict = {}
    with open(embedding_txt, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = "".join(values[:-300])
            vector = np.asarray(values[-300:], dtype="float32")
            embeddings_dict[word] = vector
    return embeddings_dict

def create_list_gold(): 
    #Change dataset here to use something else
    xsum = load_dataset('xsum', split='train')

    #Create the embedding dict used for the similarity filter
    embeddings_dict = create_embedding_dict_840()

    nb_summaries = len(xsum)
    gold_answers = []
    no_ans = 0
    one_ans = 0
    more_than_one_ans = 0

    for i in range(nb_summaries):
        answers = []
        summary_score = {}

        #Change here if you use a dataset with another structure than xsum
        passage = xsum[i]["document"]
        summary = xsum[i]["summary"]
        id_doc = xsum[i]["id"]
        tokenized_passage = word_tokenize(passage)
        tokenized_summary = word_tokenize(summary)
        tagged_summary = pos_tag(tokenized_summary)
        filtered1_answers, filtered2_answers, filtered3_answers = [], [], []

        for word in tagged_summary:
            tag = word[1]
            #Keeping only noun and verbs for subtask 2
            if tag == "NN" or tag == "VB":
                score = NonSpecificity_Scorer(passage, summary, word[0])
                #summary_score.append([word[0], score])
                summary_score[word[0]] = score
                #Filter on score
                if score < 6:
                    answers.append(word[0])
                    filtered1_answers = filter_lemma(tokenized_passage, answers)
                    filtered2_answers = filter_syn_ant(tokenized_passage, filtered1_answers)
                    filtered3_answers = filter_similarity(embeddings_dict, tokenized_passage, filtered2_answers)

        if len(filtered3_answers) == 0:
            no_ans += 1
            golden_answer = ""
        elif len(filtered3_answers) == 1:
            one_ans += 1
            golden_answer = filtered3_answers[0]
        elif len(filtered3_answers) > 1:
            more_than_one_ans += 1
            #If we have multiple possible answer after all the filter, the final golden option is the one with the lowest score
            dict_temp = {}
            for answer in filtered3_answers:
                dict_temp[answer] = summary_score[answer]
            #If same score, we pick the first one
            golden_answer = min(dict_temp, key=dict_temp.get)
        gold_answers.append(golden_answer)

    log = "After filtering {} summaries, there are :\n{} with no possible golden answer,\n{} with one possible golden answer,\n{} with more than one possible golden answer".format(nb_summaries, no_ans, one_ans, more_than_one_ans)

    return gold_answers, log

def create_list_gold_oracle_2(passages, summaries): 

    #Create the embedding dict used for the similarity filter
    embeddings_dict = create_embedding_dict_840()

    gold_answers = []
    no_ans = 0
    one_ans = 0
    more_than_one_ans = 0

    for passage, summary in zip(passages, summaries):
        answers = []
        summary_score = {}

        tokenized_passage = word_tokenize(passage)
        tokenized_summary = word_tokenize(summary)
        tagged_summary = pos_tag(tokenized_summary)
        filtered1_answers, filtered2_answers, filtered3_answers = [], [], []

        for word in tagged_summary:
            tag = word[1]
            #Keeping only noun and verbs for subtask 2
            if tag == "NN" or tag == "VB":
                score = NonSpecificity_Scorer(passage, summary, word[0])
                #summary_score.append([word[0], score])
                summary_score[word[0]] = score
                #Filter on score
                if score < 6:
                    answers.append(word[0])
                    filtered1_answers = filter_lemma(tokenized_passage, answers)
                    filtered2_answers = filter_syn_ant(tokenized_passage, filtered1_answers)
                    filtered3_answers = filter_similarity(embeddings_dict, tokenized_passage, filtered2_answers)

        if len(filtered3_answers) == 0:
            no_ans += 1
            golden_answer = ""
        elif len(filtered3_answers) == 1:
            one_ans += 1
            golden_answer = filtered3_answers[0]
        elif len(filtered3_answers) > 1:
            more_than_one_ans += 1
            #If we have multiple possible answer after all the filter, the final golden option is the one with the lowest score
            dict_temp = {}
            for answer in filtered3_answers:
                dict_temp[answer] = summary_score[answer]
            #If same score, we pick the first one
            golden_answer = min(dict_temp, key=dict_temp.get)
        gold_answers.append(golden_answer)

    log = "After filtering {} summaries, there are :\n{} with no possible golden answer,\n{} with one possible golden answer,\n{} with more than one possible golden answer".format(nb_summaries, no_ans, one_ans, more_than_one_ans)

    return gold_answers, log


def main():
    import time
    t0 = time.time()
    gold_answers, log = create_list_gold()
    with open("gold_answers_840.txt", "w+") as f:
        f.write(str(gold_answers))
    with open("log_840.txt", "w+") as f:
        f.write(log + "\n")
        f.write(f"Using glove 840, it took {time.time() - t0}s to process XSUM")

if __name__ == "__main__":
    main()

    
