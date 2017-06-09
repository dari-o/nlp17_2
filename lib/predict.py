import os

import tensorflow as tf
import numpy as np
from datetime import datetime

from lib import data_utils
from lib.seq2seq_model_utils import create_model, get_predicted_sentence, get_predicted_sentence_given_target


def predict(args, debug=False):
    def _get_test_dataset():
        with open(args.eval_path) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences

    results_filename = '_'.join(['results', str(args.pretrain_embeddings), str(args.use_attention), str(args.beam_size), str(args.antilm)])
    results_path = os.path.join(args.results_dir, results_filename+'.txt')
    avg_results_path = os.path.join(args.results_dir, results_filename+'_avg.txt')
    sentences_only_path = os.path.join(args.results_dir, results_filename+'_sent.txt')

    sentences_only_fh = open(sentences_only_path, 'w')
    avg_results_fh = open(avg_results_path, 'w')
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
            for sentence in trip_parts[:2]:
                tripID += 1
                sentenceCounter += 1
                # Get token-ids for the input sentence.
                predicted_sentence = get_predicted_sentence_given_target(args, sentence, trip_parts[tripID+1], vocab, rev_vocab, model, sess, debug=debug)

                if isinstance(predicted_sentence, list):
                    #print("%s : (%s)" % (sentence, datetime.now()))
                    #results_fh.write("%s : (%s)\n" % (sentence, datetime.now()))
                    chosenSentenceIdx = 0
                    highestProb = 0
                    i = -1
                    for sent in predicted_sentence:
                        i+=1
                        #print("  (%f)(P%f) -> %s\n" % (sent['prob'], sent['perp'], sent['dec_inp']))
                        #results_fh.write("  (%f)(P%f) -> %s\n" % (sent['prob'], sent['avgPerp'], sent['dec_inp']))
                        if( sent['prob'] > highestProb ):
                            highestProb = sent['prob']
                            chosenSentenceIdx = i
                    chosen_sentence = predicted_sentence[chosenSentenceIdx]
                else:
                    chosen_sentence = predicted_sentence


                avgPerp = chosen_sentence['perp']

                if((not (sentenceCounter % 100)) or (sentenceCounter < 10)):
                    print("%d: (P%f)%s -> %s\n" % (sentenceCounter, avgPerp, sentence, chosen_sentence['dec_inp']))
                #results_fh.write("(P%f)%s -> %s\n" % (avgPerp,sentence, chosen_sentence['dec_inp']))
                sentences_only_fh.write("%s\n" % (chosen_sentence['dec_inp']))
                # break

                perplexities.append(avgPerp)
                mmi.append(chosen_sentence['prob'])
                if(not (sentenceCounter % 5000)):
                    results_fh.flush()
                    avg_results_fh.write("intermediate: %f %f \n" % (np.mean(perplexities), np.median(perplexities)))
                    avg_results_fh.write("%f %f \n" % (np.mean(mmi), np.median(mmi)))
                    sentences_only_fh.flush()
                    results_fh.flush()


        avg_results_fh.write("%f %f " % (np.mean(perplexities), np.median(perplexities)))
        avg_results_fh.write("%f %f " % (np.mean(mmi), np.median(mmi)))

    results_fh.close()
    avg_results_fh.close()
    sentences_only_fh.close()
    print("results written in %s" % results_path)
