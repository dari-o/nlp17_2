import os

import tensorflow as tf
from datetime import datetime

from lib import data_utils
from lib.seq2seq_model_utils import create_model, get_predicted_sentence


def predict(args, debug=False):
    def _get_test_dataset():
        with open(args.test_dataset_path) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences

    results_filename = '_'.join(['results', str(args.num_layers), str(args.size), str(args.vocab_size)])
    results_path = os.path.join(args.results_dir, results_filename+'.txt')
    avg_results_path = os.path.join(args.results_dir, results_filename+'avg.txt')

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
        avgPerplexity = 0

        for sentence in test_dataset:
            sentenceCounter += 1
            # Get token-ids for the input sentence.
            predicted_sentence = get_predicted_sentence(args, sentence, vocab, rev_vocab, model, sess, debug=debug)

            if isinstance(predicted_sentence, list):
                print("%s : (%s)" % (sentence, datetime.now()))
                results_fh.write("%s : (%s)\n" % (sentence, datetime.now()))
                chosenSentenceIdx = 0
                highestProb = 0
                i = -1
                for sent in predicted_sentence:
                    i+=1
                    print("  (%f)(P%f) -> %s\n" % (sent['prob'], sent['perp'], sent['dec_inp']))
                    #results_fh.write("  (%f)(P%f) -> %s\n" % (sent['prob'], sent['avgPerp'], sent['dec_inp']))
                    if( sent['prob'] > highestProb ):
                        highestProb = sent['prob']
                        chosenSentenceIdx = i
                chosen_sentence = sentence[chosenSentenceIdx]
            else:
                chosen_sentence = predicted_sentence
            print(sentence, ' -> ', chosen_sentence)
            results_fh.write("(P%f)%s -> %s\n" % (chosen_sentence['perp'],sentence, chosen_sentence['dec_inp']))
            # break

            avgPerplexity += chosen_sentence['perp']
            if(sentenceCounter % 100):
                results_fh.flush()

        avg_results_fh.write(str(avgPerplexity/sentenceCounter))
    results_fh.close()
    avg_results_fh.close()
    print("results written in %s" % results_path)
