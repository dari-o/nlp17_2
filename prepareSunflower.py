f1 = open('data/Training_Shuffled_Dataset.txt')
f2 = open('data/cornell_nonames.txt')

new = open('works/netflix/data/train/chat.txt', 'w')
fl1 = f1.readlines()
for line in fl1:
	triple = line.replace('\n', '').split('\t')
	for triID in range(len(triple)-1):
		new.write(triple[triID] + '\n')
		new.write(triple[triID+1] + '\n')

fl2 = f2.readlines()
for line in fl2:
	triple = line.replace('\n', '').split('\t')
	for triID in range(len(triple)-1):
		new.write(triple[triID] + '\n')
		new.write(triple[triID+1] + '\n')
new.close()