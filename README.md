# Document-Similarity-Topic-Modelling

**Part 1 - Path similarity between two documents**

Functions:
* `convert_tag:` converts the tag given by `nltk.pos_tag` to a tag used by `wordnet.synsets`. This function is used in `doc_to_synsets`.
* `document_path_similarity:` computes the symmetrical path similarity between two documents by finding the synsets in each document using `doc_to_synsets`, then computing similarities using `similarity_score`.

* `doc_to_synsets:` returns a list of synsets in document. This function first tokenizes and part of speech tags the document using `nltk.word_tokenize` and `nltk.pos_tag`. Then it finds each tokens corresponding synset using `wn.synsets(token, wordnet_tag)`. The first synset match is used. If there is no match, that token is skipped.
* `similarity_score:` returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2). For each synset in s1, it finds the synset in s2 with the largest similarity value. Sums all of the largest similarity values together and normalizes this value by dividing it by the number of largest similarity values found. Missing values are ignored.


**Part 2 - Topic Modelling**

Gensim's LDA (Latent Dirichlet Allocation) model is used to model topics in `newsgroup_data`. Using gensim.models.ldamodel.LdaModel constructor, LDA model parameters are estimated on the corpus, and saved to the variable `ldamodel`. 10 topics are extracted using `corpus` and `id_map`, and with `passes=25` and `random_state=34`.
