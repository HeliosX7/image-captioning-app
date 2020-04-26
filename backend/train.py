from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint


def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text
 
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

def load_clean_descriptions(filename, dataset):
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		image_id, image_desc = tokens[0], tokens[1:]
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id] = list()
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			descriptions[image_id].append(desc)
	return descriptions

def load_photo_features(filename, dataset):
	all_features = load(open(filename, 'rb'))
	features = {k: all_features[k] for k in dataset}
	return features



def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 

def create_sequences(tokenizer, max_length, desc_list, photo):
	X1, X2, y = list(), list(), list()
	for desc in desc_list:
		seq = tokenizer.texts_to_sequences([desc])[0]
		for i in range(1, len(seq)):
			in_seq, out_seq = seq[:i], seq[i]
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)

def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

def define_model(vocab_size,max_length):
	inputs1=Input(shape=(4096,))
	fe1=Dropout(0.5)(inputs1)
	fe2=Dense(256,activation='relu')(fe1)

	inputs2=Input(shape=(max_length,))
	se1=Embedding(vocab_size,256,mask_zero=True)(inputs2)
	se2=Dropout(0.5)(se1)
	se3=LSTM(256)(se2)

	decoder1=add([fe2,se3])
	decoder2=Dense(256,activation='relu')(decoder1)
	outputs=Dense(vocab_size,activation='softmax')(decoder2)

	model=Model(inputs=[inputs1,inputs2],outputs=outputs)
	model.compile(loss='categorical_crossentropy',optimizer='adam')

	print(model.summary())
	#plot_model(model,to_file='model.png',show_shapes=True)
	return model

###################################
def data_generator(descriptions, photos, tokenizer, max_length):
	while 1:
		for key, desc_list in descriptions.items():
			photo = photos[key][0]
			in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
			yield [[in_img, in_seq], out_word]

##################################

filename = 'Flickr_8k.trainImages.txt'
train = load_set(filename)
print("train len pics: ",len(train))
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print("desc len: ",len(train_descriptions))
train_features = load_photo_features('features.pkl', train)
print("length of pic features: ",len(train_features))
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print("vocab size: ", vocab_size)
max_length = max_length(train_descriptions)
print("maxlen: ",max_length)
#print("yyyyy")
 
model = define_model(vocab_size, max_length)
epochs = 500
steps = len(train_descriptions)/32
for i in range(epochs):
	generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
	model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
	model.save('model_' + str(i) + '.h5')

