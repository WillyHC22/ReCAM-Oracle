This repository is used for replicating ReCAM dataset (Reading Comprehension of Abstract Meaning) used for subtask 2 of SemEval 2021 Task 4: [paper](https://arxiv.org/pdf/2105.14879.pdf)

## Objective

The ReCAM dataset is built using the XSUM dataset as a base, an abstractive summarization dataset. In each data sample, we find abstract concepts following two definition of abstractness: 1) Imperceptibility and 2) Non-Specificity. Imperceivable concepts refers to non-tangible ideas that are distant from immediate perception (objective, culture, economy). Non-Specific concepts refers to broad denomination of a category, for example a "vertebrate" is a more abstract concept than a "whale" or a "dog".   

After finding the abstract concepts following each definition, we generate replace the abstract token by a placeholder, then generate a question and 4 pseudo-answers that could fill the new placeholder in order to convert the summarization dataset into a MCQA dataset.

## Steps

Two general steps (Not doing second one because we are using hte same way TA-MAMC chose to generate pseudo answers):

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
