#this should be called by main with the 'eval' mode, and produce the ouput required in the project description
import os

import tensorflow as tf
import numpy as np
from datetime import datetime

from lib import data_utils
from lib.seq2seq_model_utils import create_model, get_predicted_sentence, get_predicted_sentence_given_target


def evaluate(args, debug=False):
    def _get_test_dataset():
        with open(args.eval_path) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences

    results_filename = '_'.join(['results', str(args.pretrain_embeddings), str(args.use_attention), str(args.beam_size), str(args.antilm)])
    results_path = os.path.join(args.results_dir, results_filename+'.txt')

    with tf.Session() as sess, open(results_path, 'w') as results_fh:
        # Create model and load parameters.
        args.batch_size = 1
        model = create_model(sess, args)

        # Load vocabularies.
        vocab_path = os.path.join(args.data_dir, "vocab%d.in" % args.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path, args.dict_path)

        test_dataset = _get_test_dataset()

        sentenceCounter = -1
        perplexities = []
        mmi = []

        if not args.antilm:
            mmi = [0]

        for trip in test_dataset:
            trip_parts = trip.split('\t')
            tripID = -1
            oldAvgPerp = 0
            for sentence in trip_parts[:2]:
                tripID += 1
                sentenceCounter += 1
                # Get token-ids for the input sentence.
                predicted_sentence = get_predicted_sentence_given_target(args, sentence, trip_parts[tripID+1], vocab, rev_vocab, model, sess, debug=debug)

                if isinstance(predicted_sentence, list):
                    chosenSentenceIdx = 0
                    highestProb = 0
                    i = -1
                    for sent in predicted_sentence:
                        i+=1
                        if( sent['prob'] > highestProb ):
                            highestProb = sent['prob']
                            chosenSentenceIdx = i
                    chosen_sentence = predicted_sentence[chosenSentenceIdx]
                else:
                    chosen_sentence = predicted_sentence

                avgPerp = chosen_sentence['perp']

                if(tripID == 1):
                    print("%f %f" % (oldAvgPerp, avgPerp))

                oldAvgPerp = avgPerp


    results_fh.close()
