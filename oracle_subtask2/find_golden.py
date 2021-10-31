import numpy as np
from pywsd.lesk import adapted_lesk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from datasets import load_dataset 

from utils import filter_lemma, filter_syn_ant, filter_similarity


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



def create_embedding_dict(embedding_txt="glove.6B.50d.txt"):
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



def main(): 
    #Change dataset here to use something else
    xsum = load_dataset('xsum', split='train')

    #Create the embedding dict used for the similarity filter
    embeddings_dict = create_embedding_dict("glove.6B.50d.txt")

    #If we want to process the whole dataset:
    #nb_summaries = len(xsum)

    nb_summaries = 100
    no_ans = 0
    one_ans = 0
    more_than_one_ans = 0

    for i in range(nb_summaries):
        answers = []
        #summary_score = []
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

        f.write("Summary used : {} \n".format(summary))
        f.write("id of the summary : {} \n".format(id_doc))
        f.write("{} \n".format(summary_score))
        f.write("First batch of answers after NS Score filtering : {} \n".format(answers))
        f.write("Second batch of answers after Lemma filter : {} \n".format(filtered1_answers))
        f.write("Third batch of answers after Synonym/Antomym filter : {} \n".format(filtered2_answers))
        f.write("Fourth batch of answers after Similarity filter : {} \n".format(filtered3_answers))
        f.write("The golden answer kept for this summary is : {} \r\n".format(golden_answer))
    
    print("After filtering {} summaries, there are :\n{} with no possible golden answer,\n{} with one possible golden answer,\n{} with more than one possible golden answer".format(nb_summaries, no_ans, one_ans, more_than_one_ans))



if __name__ == "__main__":

    with open("output_oracle_subtask2.txt", "w+") as f:
        main()
