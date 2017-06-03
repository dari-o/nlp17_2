f = open('data/Training_Shuffled_Dataset.txt')
new = open('../tf_chatbot_seq2seq_antilm/works/netflix/data/train/chat.txt', 'w')
f = f.readlines()

for line in f:
	triple = line.replace('\n', '').split('\t')
	new.write(triple[0] + '\n')
	new.write(triple[1] + '\n')
	new.write(triple[1] + '\n')
	new.write(triple[2] + '\n')
new.close()