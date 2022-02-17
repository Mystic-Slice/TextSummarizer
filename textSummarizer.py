from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re

INPUT_FILE = "text.txt"
OUTPUT_FILE = "summary.txt"
REDUCED_SIZE = 0.3

def readAndCleanSentences(fileName):
    # Read lines
    with open(fileName, "r") as f:
        sentences = [line.strip() for line in ". ".join(f.readlines()).split(". ")]
    
    # Remove special characters and split into words
    sentences = [re.sub(r"[^a-zA-Z0-9\(\)\[\]]", " ", line) for line in sentences]
    sentences = [line.split() for line in sentences if line.split()]
    return sentences

def sentenceSimilarity(sentence1, sentence2, stopWords=None):
    if stopWords is None:
        stopWords = []

    # checking membership in set is slightly faster than in list
    stopWords = set(stopWords)

    # convert to lower case
    sentence1 = [word.lower() for word in sentence1]
    sentence2 = [word.lower() for word in sentence2]

    # create a list of all words to build the word vector
    allWords = list(set(sentence1 + sentence2))

    # word vector...each index corresponds to a word
    wordVector1 = np.zeros(len(allWords))
    wordVector2 = wordVector1.copy()

    # count of the word in the sentence....0 if not present
    for word in sentence1:
        if word in stopWords: continue
        wordVector1[allWords.index(word)] += 1
    
    for word in sentence2:
        if word in stopWords: continue
        wordVector2[allWords.index(word)] += 1

    # cosine distance is 0 when the vectors overlap (basically the same)
    # so 1-cosine distance gives value of 1 denoting that they are very similar
    # and vice versa

    # cosine similarity = cos(theta) (measure of how close they are)
    # cosine distance = 1 - cosine similarity (measure of how far apart they are)
    return 1-cosine_distance(wordVector1, wordVector2)

def compareSentences(sentences, stopWords):
    # creating 2-D similarity matrix with default value 1
    similarityMatrix = np.array([np.ones(len(sentences)) for i in range(len(sentences))])

    for i, sentence1 in enumerate(sentences):
        for j, sentence2 in enumerate(sentences):
            if i==j:
                # same sentences will have similarity 1
                continue
            similarityMatrix[i][j] = sentenceSimilarity(sentence1, sentence2, stopWords)

    return similarityMatrix

def summarize(fileName, reducedSize):
    if not 0 < reducedSize < 1:
        reducedSize = 0.5

    stopWords = stopwords.words("english")

    # read from file
    sentences = readAndCleanSentences(fileName)
    numSentences = round(len(sentences) * reducedSize)

    # create similarity matrix
    similarityMatrix = compareSentences(sentences, stopWords)

    # compute ranking of nodes based on similarity with other nodes
    # basically sentences that are similar to most other sentences have higher ranking
    scores = nx.pagerank(nx.from_numpy_array(similarityMatrix))

    # pick the first n sentences
    rankedSentences = sorted([(scores[i], s) for i, s in enumerate(sentences)], reverse=True)

    # create the summary
    summary = ". ".join([" ".join(rankedSentences[i][1]) for i in range(numSentences)])
    return summary + '.'

with open(OUTPUT_FILE, "w") as f:
    print(summarize(INPUT_FILE, REDUCED_SIZE), file=f)
