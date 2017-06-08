import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from lib.load_embeddings import load_embedding
from lib.Constants import *
from numpy import linalg as LA
import pdb


#Compute greedy similaryty
model_output_path = 'results_4_300_20000sent.txt'
ground_truth_path = 'validation_ground_truth.txt'
word2int_path = "works" + sep +"netflix" +sep+ "data"+ sep+"word2int_dictionary.npy"
word2int = np.load(pathToWord2Int).item()
model_output = open(model_output_path).readlines()
ground_truth = len(model_output)
vocabSize = 20000

embeddingSize=512

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
def cosine_similarity(u, w):
	u_norm = LA.norm(u)
	w_norm = LA.norm(w)
	return np.dot(u,w)/(u_norm*w_norm)



def sent2ints(sent):
	int_sent = np.zeros(len(sent))
	for w in range(len(sent)):
		if sent[w] in word2int:
			int_sent[w] = word2int[sent[w]]
		else:
			int_sent[w] = UNK_ID
	return int_sent

def get_max(w, r2):
	max_sim = -1
	for w2 in r2:
		cos_sim = cosine_similarity(w, w2)
		if cos_sim > max_sim:
			max_sim = cos_sim
	return max_sim

def G(r1, r2):
	sum = 0
	for w in range(len(r1)):
		sum += get_max(r1[w], r2)
	return sum/len(r1)

def GM(r1, r2):
	return (G(r1, r2)+G(r2,r1))/2


def average_greedy(model_output_emb, ground_truth_emb):	
	sum = 0
	for i in range(num_responses):
		sum += GM(model_output_emb[i], ground_truth_emb[i])
	return sum/num_responses

session = tf.Session()
model_output_emb = []
ground_truth_emb = []

weightsEmbedding    = tf.Variable(tf.random_uniform([vocabSize, embeddingSize], -0.1, 0.1), name="Wembedding")
load_embedding(session, weightsEmbedding)

initialization = tf.global_variables_initializer()
session.run(initialization)
with session.as_default():
	for i in range(num_responses):
		#print(i)
		out_sent_int = sent2ints(model_output[i].split())
		truth_sent_int = sent2ints(ground_truth[i].split())
		#pdb.set_trace()
		x  = tf.nn.embedding_lookup( weightsEmbedding, tf.to_int32(out_sent_int))
		x2 = tf.nn.embedding_lookup( weightsEmbedding, tf.to_int32(out_sent_int))
		model_output_emb.append(x.eval())
		ground_truth_emb.append(x2.eval())
	print("done")

eval = average_greedy(model_output_emb, ground_truth_emb)
print("CHECK PATHS AND NUM_RESPONSES FOR THE NUMBER OF SENTANCES WE WANT TO EVALUATE")
print(eval)
session.close()