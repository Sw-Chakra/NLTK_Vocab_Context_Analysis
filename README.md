# NLTK_Vocab_Context_Analysis
You shall know a word by the company it keeps.

The large number of English words can make language-based applications daunting. To cope with this, it is
helpful to have a clustering or embedding of these words, so that words with similar meanings are clustered
together, or have embeddings that are close to one another.

Words that tend to appear in similar contexts are likely to be related. In this Project, we would
investigate this idea by coming up with an embedding of words that is based on co-occurrence statistics.


** Reference: CSE 250 Git Repositories

STEPS:

 First, download the Brown corpus (using nltk.corpus). This is a collection of text samples from a
wide range of sources, with a total of over a million words. Calling brown.words() returns this text
in one long list, which is useful.

 Remove stopwords and punctuation, make everything lowercase, and count how often each word occurs.
Use this to come up with two lists:

{ A vocabulary V , consisting of a few thousand (e.g., 5000) of the most commonly-occurring words.
{ A shorter list C of at most 1000 of the most commonly-occurring words, which we shall call
context words.

 For each word w in V , and each occurrence of it in the text stream, look at the surrounding window of
four words (two before, two after): w1 w2 w w3 w4:
Keep count of how often context words from C appear in these positions around word w.

Using these counts, construct the probability distribution Pr(c|w) of context words around w (for each
w in V ), as well as the overall distribution Pr(c) of context words. These are distributions over C.
Compute the (positive) pointwise mutual information on word embeddings.

 Perform dimensionality reduction of the word vectors and categorize them into word clusters

 Word clusters are analyzed and nearest neighbours within the clusters are analyzed to understand how well the algorithm has performed.
