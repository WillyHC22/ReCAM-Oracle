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
