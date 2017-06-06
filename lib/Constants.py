import os

BUILD_DICT = False#should be true whenever the max sentence length changes - but has to be the same for en-and decoding runs !!

DEV_DATA = 400000
MAX_SENTENCE_LEN = 100
USE_EMBEDDINGS = False
UNK = "<unk>"
PAD = "<pad>"
EOS = "<eos>"
BOS = "<bos>"

embedding_size = 512
MAX_SENTENCE_LEN = 100
attention = True

sep = os.sep
buckets =  [(5, 10), (10, 15), (20, 25), (40, 50),(50,100)]

workingDirectory    = os.getcwd()
pathToData = workingDirectory+sep+"works" + sep +"netflix" +sep+ "data"
pathToTrainData     = pathToData+sep+"Training_Shuffled_Dataset.txt"
pathToEvalData      = pathToData+sep+"Validation_Shuffled_Dataset.txt"
pathToTrainLabels   = pathToData+sep+"Training_Shuffled_Dataset_Labels.txt"
pathToEvalLabels    = pathToData+sep+"Validation_Shuffled_Dataset_Labels.txt"
pathToCornellData   = pathToData+sep+"cornell_nonames.txt"

pathToEmbeddings     = pathToData+sep+"embeddings"+ sep + "MIN5Model_+Cornell_size"+str(embedding_size)+".word2vec"
pathToModel         = workingDirectory+sep+sep+"model"
pathToWord2Int      = pathToData + sep+"word2int_dictionary.npy"
pathToIntData       = workingDirectory + sep + "integrizedMatrix.npy"
pathToIntTrainDataIn = pathToTrainData+"I.in"
pathToIntTrainDataOut = pathToTrainData+"I.out"
pathToIntTrainDataInDev = pathToTrainData+"Idev.in"


pathToIntTrainDataOutDev = pathToTrainData+"Idev.out"
