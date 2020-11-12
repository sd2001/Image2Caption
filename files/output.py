import modules as m

def extract_features(file):
    model=m.VGG16()
    model=m.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image=m.load_img(file, target_size=(224, 224))
    image=m.img_to_array(image)
    image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image=m.preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def extract_features2(img):
    model=m.VGG16()
    model=m.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image=img.resize((224, 224))
    image=m.img_to_array(image)
    image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image=m.preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def word2id(integer,tokenizer):
    for word,i in tokenizer.word_index.items():
        if i==integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = m.pad_sequences([sequence], maxlen=max_length)
        ypred=model.predict([photo,sequence], verbose=0)
        ypred=m.argmax(ypred)
        word=word2id(ypred, tokenizer)
        if word is None:
            break
        in_text+=' '+word
        if word == 'endseq':
            break
    return in_text

tokenizer=m.load(open('tokenizer.pkl', 'rb'))
max_length=34
model=m.load_model('model_9.h5')
pic='test14.jpg'
photo=extract_features(pic)
description=generate_desc(model, tokenizer, photo, max_length)

im = m.array(m.Image.open(pic))
m.plt.imshow(im)
print(description)
del generate_desc,extract_features