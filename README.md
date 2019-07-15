# nlp-study-group
Notes on natural language processing.

# Projects

## Current project #1 Detecting bias in Wikipedia page edit

A user enters a URL for a wikipedia page. The history is retrieved and sentiment analysis is conducted on each edit to determine what, if any, bias change is detected. The following refereces are guiding the project:



- [seq2seq](https://github.com/google/seq2seq): A general-purpose encoder-decoder framework for Tensorflow that can be used for Machine Translation, Text Summarization, Conversational Modeling, Image Captioning, and more.

- [Open NMT](http://opennmt.net/): OpenNMT is an open source ecosystem for neural machine translation and neural sequence learning.

- [Harvard NLP code](http://nlp.seas.harvard.edu/code/): Includes seq2seq , Open NMT, Attention models.

- [Attention is all you need](https://arxiv.org/abs/1706.03762): The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

- [The annotated Transformer from Attention is all you need](http://nlp.seas.harvard.edu/2018/04/03/attention.html):In this post I present an “annotated” version of the paper in the form of a line-by-line implementation. I have reordered and deleted some sections from the original paper and added comments throughout. This document itself is a working notebook, and should be a completely usable implementation. In total there are 400 lines of library code which can process 27,000 tokens per second on 4 GPUs.

- [Text Blob](https://textblob.readthedocs.io/en/dev/quickstart.html): TextBlob aims to provide access to common text-processing operations through a familiar interface. You can treat TextBlob objects as if they were Python strings that learned how to do Natural Language Processing. Good for sentiment analysis.

- [Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews](https://nrc-publications.canada.ca/nparc/eng/view/object/?id=4bb7a0c8-9d9b-4ded-bcf6-fdf64ee28ccc)

- [Global Vectors for Word Representation (GLoVe)](https://github.com/stanfordnlp/GloVe): e provide an implementation of the GloVe model for learning word representations, and describe how to download web-dataset vectors or train your own. See the project page or the paper for more information on glove vectors.

## Current project #2 CS224n Assignment 2 
Currently studying Winter 2017 CS224n Assignment #2.
 - Colaboratory with tensorboard in progress
   - [rymc9384/DeepNLP_CS224N](https://github.com/rymc9384/DeepNLP_CS224N)
   - [2017 Natural Language Processing with Deep Learning Video Playlist](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)
   - [2019 assigments in PyTorch](http://web.stanford.edu/class/cs224n/index.html#coursework)

# Papers

- [Word embedding summary](http://ruder.io/word-embeddings-1/): This first post lays the foundations by presenting current word embeddings based on language modelling. While many of these models have been discussed at length, we hope that investigating and discussing their merits in the context of past and current research will provide new insights.

- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781): This is **word2vec**. We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.

- [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf): by Quoc Le and Tomas Mikolov. (This is **doc2vec**). Many machine learning algorithms require the
input to be represented as a fixed-length feature
vector. When it comes to texts, one of the most
common fixed-length features is bag-of-words.
Despite their popularity, bag-of-words features
have two major weaknesses: they lose the ordering
of the words and they also ignore semantics
of the words. For example, “powerful,” “strong”
and “Paris” are equally distant. In this paper, we
propose Paragraph Vector, an unsupervised algorithm
that learns fixed-length feature representations
from variable-length pieces of texts, such as
sentences, paragraphs, and documents. Our algorithm
represents each document by a dense vector
which is trained to predict words in the document.
Its construction gives our algorithm the
potential to overcome the weaknesses of bag-ofwords
models. Empirical results show that Paragraph
Vectors outperform bag-of-words models
as well as other techniques for text representations.
Finally, we achieve new state-of-the-art results
on several text classification and sentiment
analysis tasks

- [LDAvis: A method for visualizing and interpreting topics](https://www.aclweb.org/anthology/W14-3110) by Carson Sievert and Kenneth E. Shirley. We present LDAvis, a web-based interactive
visualization of topics estimated using
Latent Dirichlet Allocation that is built using
a combination of R and D3. Our visualization
provides a global view of the topics
(and how they differ from each other),
while at the same time allowing for a deep
inspection of the terms most highly associated
with each individual topic. First,
we propose a novel method for choosing
which terms to present to a user to aid in
the task of topic interpretation, in which
we define the relevance of a term to a
topic. Second, we present results from a
user study that suggest that ranking terms
purely by their probability under a topic is
suboptimal for topic interpretation. Last,
we describe LDAvis, our visualization
system that allows users to flexibly explore
topic-term relationships using relevance to
better understand a fitted LDA model.

# Links
- [PyTorch at Udemy](https://github.com/udacity/deep-learning-v2-pytorch)
- [awesome-nlp](https://github.com/keon/awesome-nlp) - goog collection of NLP links
- [NLP Overview](https://nlpoverview.com/) - Modern Deep Learning Techniques Applied to Natural Language Processing 
- [NLP Progress - Sebastian Ruder](https://nlpprogress.com/) - Current state-of-the-art for the most common NLP tasks
- [NLP Blog - Sebasian Ruder](http://ruder.io/word-embeddings-1/) - Good explanation of NLP history and methodology
- [Fuzzy C-means clustering in R](https://cran.r-project.org/web/packages/ppclust/vignettes/fcm.html) Clustering example in R using Iris data.

# Slides
- [NMF and clustering slides from Stanford](https://web.stanford.edu/group/mmds/slides2012/s-park.pdf)
- [NMF and PCA slides](http://ranger.uta.edu/~chqding/PCAtutorial/PCA-tutor3.pdf)
- [Andrew Moore from Carnegie Mellon](https://www.autonlab.org/tutorials)

# Tools
- [AllenNLP](https://allennlp.org/) High powered NLP library.

# Books
- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James H. Martin. Nice 
looking, covers n-grams, naive bayes classifiers, sentiment, logistic regression, vector semantics, neural nets,
part-of-speech tagging, sequence processing with recurrent networks, grammers, syntax, statistical parsing, information
extraction. hidden markov models.

# Wikipedia
- https://en.wikipedia.org/wiki/Natural_language_processing
- https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
- https://en.wikipedia.org/wiki/Graph_theory
- https://en.wikipedia.org/wiki/Bayesian_network
- https://en.wikipedia.org/wiki/Computational_linguistics
- https://en.wikipedia.org/wiki/Fuzzy_clustering
