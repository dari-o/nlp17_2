from gensim import models
import numpy as np
import tensorflow as tf
import pdb 
from lib.Constants import *
def load_embedding(session, emb):
    '''
      session        Tensorflow session object
      vocab          A dictionary mapping token strings to vocabulary IDs
      emb            Embedding tensor of shape vocabulary_size x dim_embedding
      path           Path to embedding file
      dim_embedding  Dimensionality of the external embedding.
    '''
    print("Loading external embeddings from %s" % path)
    vocab = np.load(pathToWord2Int).item()
    vocab_size = len(vocab)
    model = models.KeyedVectors.load_word2vec_format(pathToEmbeddings, binary=False)
    external_embedding = np.zeros(shape=(vocabSize, embedding_size))
    matches = 0
    for tok, idx in vocab.items():
        if tok in model.vocab:
            external_embedding[idx] = model[tok]
            matches += 1
        else:
            print("%s not in embedding file" % tok)
            external_embedding[idx] = np.random.uniform(low=-0.25, high=0.25, size=dim_embedding)
        
    print("%d words out of %d could be loaded" % (matches, vocabSize))
    
    pretrained_embeddings = tf.placeholder(tf.float32, [None, None]) 
    assign_op = emb.assign(pretrained_embeddings)
    session.run(assign_op, {pretrained_embeddings: external_embedding})
    return pretrained_embeddings
