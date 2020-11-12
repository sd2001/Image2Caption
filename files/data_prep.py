import modules as m
import model as mo
import data_prep_utils as u

txt=u.make_txt(m.IMG_CAPTION)
descriptions=u.make_dict(txt)
u.clean_dict(descriptions)
vocabulary=u.vocab_create(descriptions)
u.save_descriptions(descriptions)

test=u.make_set(m.TEST_IMG_NAME)
test_descriptions=u.load_clean_dict(test)
test_features=u.load_photos('features.pkl',test)

train=u.make_set(m.TRAIN_IMG_NAME)
#print(len(train))
train_descriptions=u.load_clean_dict(train)
#print(len(train_descriptions))
train_features=u.load_photos('features.pkl',train)
#print(len(train_features))

tokenizer=u.create_tokenizer(train_descriptions)
vocab_size=len(tokenizer.word_index) + 1
#print('Vocabulary Size: %d' % vocab_size)
maxlen=u.max_length(train_descriptions)

train=u.make_set(m.TRAIN_IMG_NAME)
#print('Dataset: %d' % len(train))
train_descriptions=u.load_clean_dict(train)
#print('Descriptions: train=%d' % len(train_descriptions))
train_features=u.load_photos('features.pkl',train)
#print('Photos: train=%d' % len(train_features))
vocab_size=len(tokenizer.word_index) + 1

model=mo.define_model(vocab_size, maxlen)
### Since I already trained this, I am loading a model instead of training once again
model=m.load_model('model_9.h5')
epochs=10
steps=len(train_descriptions)
### Do uncomment if the model is to be trained again
#for i in range(epochs):
    #generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)

    #model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    #model.save('model_' + str(i) + '.h5')