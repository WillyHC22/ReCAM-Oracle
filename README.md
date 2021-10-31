This repository is used for replicating ReCAM dataset used for subtask 2 of SemEval 2021 Task 4.

https://arxiv.org/pdf/2105.14879.pdf

The dataset is built using XSUM dataset as a base. We first find the abstract concept in the summaries provided from XSUM dataset, then we replace then by a placeholder. We then generate 4 other options in order to transform the XSUM dataset into a Reading Comprehension of Abstract Meaning (ReCAM) dataset.


Two general steps :

1) Find non specific concept :
- Abstractness is based on WordNet hypernyms hierarchy in terms of distance to the root (0-17). They keep words with distance <6 to root. They also keep only nouns/verbs.
- They use Adapted Lesk Algorithm for each token in the concatenated summary/passage for sense desambiguition.

2) Construct the others options after finding the gold answer :
- They use 3 models : GA Reader, Att Reader and Att Reader + gloss to generate top 10 options for each. 
- They use the synonyms and Similarity filters (any word with more than 0.85 score for similarity with golden answer are rejected)
- Select top 4 most frequent to make it a 5-choice QA with passage.

Filters :
- Filtering by Lemmas 
- Filtering by Synonyms and antonyms
- Filtering by Similarity   