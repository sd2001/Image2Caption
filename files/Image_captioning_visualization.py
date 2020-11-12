#!/usr/bin/env python
# coding: utf-8

# # Auto Image Captioning

# In[1]:


import os
from pandas import DataFrame
from tensorflow.keras.preprocessing import image, sequence
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import ssl
import plotly
from PIL import Image
from pickle import dump,load
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Dropout,add
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model,load_model
from nltk.translate.bleu_score import corpus_bleu
from numpy import argmax
import ssl
from os import listdir


# In[2]:


IMG_DIR="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_Dataset/Flicker8k_Dataset"
TRAIN_IMG_NAME="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr_8k.trainImages.txt"
TEST_IMG_NAME="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr_8k.testImages.txt"
VALID_IMG_NAME="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr_8k.devImages.txt"
IMG_CAPTION="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr8k.token.txt"
LEMM_CAPTION="/home/sd2001/Desktop/Programming/Image Captioning/Flickr8k_text/Flickr8k.lemma.token.txt"


# ## Data Visualization and Understanding the Dataset

# In[3]:


captions=open(IMG_CAPTION, 'r').read().split("\n")
x_train=open(TRAIN_IMG_NAME, 'r').read().split("\n")
x_val=open(VALID_IMG_NAME, 'r').read().split("\n")
x_test=open(TEST_IMG_NAME, 'r').read().split("\n")


# In[4]:


c=captions[0].split('#')
c[1]


# In[5]:


print("Image name :",c[0],"| Caption :",c[1][2:])


# In[6]:


img=[]
corpus=[]
ic={}
combined=[]
for c in range(len(captions)-1):
    a=captions[c].split('#')
    image=a[0]
    cp='Start '+a[1][2:]+' End'
    combined.append([image,cp])
    img.append(image)
    corpus.append(cp)
    if image in ic:
        ic[image].append(a[1][2:])
    else:
        ic[image] = [a[1][2:]]
combined


# In[7]:


combined_df=DataFrame(combined,columns=['Image','Caption'])
ds=combined_df.values


# In[8]:


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')


# In[9]:


final_corpus=[]
dup_corpus=[]
for sent in corpus:
    words=word_tokenize(sent)
    
    for w in words:
        w=w.lower()
        if w=='.' or w=='!' or w==",":
            continue
        else:
            dup_corpus.append(w)
            if w in final_corpus:
                continue
            else:
                final_corpus.append(w)

final_corpus
        
    


# In[10]:


fdist1=nltk.FreqDist(dup_corpus)
#Get 50 Most Common Words
print (fdist1.most_common())


# In[11]:


fd=fdist1.most_common()
words=[]
aa=[]
for i in range(len(fd)):
    aa=[]
    aa.append(fd[i][0])
    aa.append(fd[i][1])
    words.append(aa)
words


# In[12]:


df=DataFrame(words,columns=['Words','Count'])
#df=DataFrame(count, columns=['Count'])


# In[13]:


df


# In[14]:


import plotly.express as px
fig = px.bar(df[:50], x='Words', y='Count',color="Count",title="Most freq occuring words")
fig.update_layout(
    font_family="Courier New",
    title_x=0.5,
    font_color="green",
    title_font_family="Times New Roman",
    title_font_color="black",
    legend_title_font_color="green"
)
fig.show()


# ## Preparing Data for our Model

# In[15]:


def make_txt(file):
    f=open(file,'r')
    text=f.read()
    f.close()
    return text
txt=make_txt(IMG_CAPTION)
txt


# In[16]:


def form_dict(img_list):
    img_dict={}
    for im in img_list:
        if im in ic:
            img_dict[im]=ic[im]
    return img_dict
form_dict(x_train)


# In[17]:


def make_dict(txt):
    x={}
    for line in txt.split('\n'):
        tokens=line.split()
        if len(line)<3:
            continue
        image_id,image_desc=tokens[0],tokens[1:]
        image_id=image_id.split('.')[0]
        image_desc = ' '.join(image_desc)
        if image_id not in x:
            x[image_id]=list()
        x[image_id].append(image_desc)
    return x
    
descriptions=make_dict(txt)
descriptions


# In[18]:


import string
def clean_dict(des):
    table=str.maketrans('', '', string.punctuation)
    for key,desc_list in des.items():
        for i in range(len(desc_list)):
            desc=desc_list[i]
            desc=desc.split()
            desc=[word.lower() for word in desc]
            desc=[w.translate(table) for w in desc]
            desc=[word for word in desc if len(word)>1]
            desc=[word for word in desc if word.isalpha()]
            desc_list[i]= ' '.join(desc)
 
# clean descriptions
clean_dict(descriptions)
descriptions


# In[19]:


def vocab_create(des):
    all=set()
    for k in des.keys():
        [all.update(d.split()) for d in des[k]]
    return all
vocabulary=vocab_create(descriptions)
len(vocabulary)


# In[20]:


def save_descriptions(descriptions):
    lines=[]
    for key,d in descriptions.items():
        for desc in d:
            lines.append(key + ' ' + desc)
    data='\n'.join(lines)
    file=open('descriptions.txt', 'w')
    file.write(data)
    file.close()
    
save_descriptions(descriptions)


# In[21]:


def make_set(file):
    txt=make_txt(file)
    d=[]
    for line in txt.split('\n'):
        if len(line)<1:
            continue
        sent=line.split('.')[0]
        d.append(sent)
    return set(d)


# In[22]:


def load_clean_dict(dataset):
    doc=make_txt('descriptions.txt')
    descriptions=dict()
    for line in doc.split('\n'):
        tokens=line.split()
        image_id,image_desc=tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc='startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions


# In[23]:


def load_photos(file,dataset):    
    filename='descriptions.txt'
    all_features=load(open(file, 'rb'))
    f={k: all_features[k] for k in dataset}
    return f


# In[24]:


test=make_set(TEST_IMG_NAME)
print(len(test))
test_descriptions=load_clean_dict(test)
print(len(test))
test_features=load_photos('features.pkl',test)


# In[25]:


train=make_set(TRAIN_IMG_NAME)
print('Dataset:%d' % len(train))
train_descriptions=load_clean_dict(train)
print('Descriptions:train=%d' % len(train_descriptions))
train_features=load_photos('features.pkl',train)
print('Photos:train=%d' % len(train_features))


# In[26]:


def max_length(des):
    lines=to_lines(des)
    return max(len(d.split()) for d in lines)


# In[27]:


def to_lines(des):
    all=[]
    for key in des.keys():
        [all.append(d) for d in des[key]]
    return all

def create_tokenizer(des):
    lines=to_lines(des)
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
 
tokenizer=create_tokenizer(train_descriptions)
vocab_size=len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# In[28]:


maxlen=max_length(train_descriptions)
maxlen


# In[29]:


len(test_descriptions),len(train_descriptions)


# In[30]:


def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1,X2,y=[],[],[]
    for desc in desc_list:
        seq=tokenizer.texts_to_sequences([desc])[0]
        for i in range(1,len(seq)):
            in_seq,out_seq=seq[:i], seq[i]
            in_seq=pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq=to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


# In[31]:


def define_model(vocab_size, max_length):
    inputs1=Input(shape=(4096,))
    fe1=Dropout(0.5)(inputs1)
    fe2=Dense(256, activation='relu')(fe1)
    inputs2=Input(shape=(max_length,))
    se1=Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2=Dropout(0.5)(se1)
    se3=LSTM(256)(se2)
    decoder1=add([fe2, se3])
    decoder2=Dense(256, activation='relu')(decoder1)
    outputs=Dense(vocab_size, activation='softmax')(decoder2)
    model=Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# In[32]:


def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    while 1:
        for key,desc_list in descriptions.items():
            photo=photos[key][0]
            in_img,in_seq, out_word=create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield ([in_img,in_seq],out_word)


# In[33]:


train=make_set(TRAIN_IMG_NAME)
print('Dataset: %d' % len(train))
train_descriptions=load_clean_dict(train)
print('Descriptions: train=%d' % len(train_descriptions))
train_features=load_photos('features.pkl',train)
print('Photos: train=%d' % len(train_features))
vocab_size=len(tokenizer.word_index) + 1
print(maxlen,vocab_size)


# In[34]:


model=define_model(vocab_size, maxlen)
#Since I already trained this, I am loading a model instead of training once again
model=load_model('model_9.h5')
epochs=10
steps=len(train_descriptions)
#Do uncomment if the model is to be trained again
#for i in range(epochs):
    #generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)

    #model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

    #model.save('model_' + str(i) + '.h5')


# In[35]:


tokenizer=create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))


# In[36]:


tokenizer = load(open('tokenizer.pkl', 'rb'))


# In[37]:


def extract_features(file):
    model=VGG16()
    model=Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image=load_img(file, target_size=(224, 224))
    image=img_to_array(image)
    image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image=preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature


# In[38]:


def word2id(integer,tokenizer):
    for word,i in tokenizer.word_index.items():
        if i==integer:
            return word
    return None


# In[44]:


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=maxlen)
        ypred=model.predict([photo,sequence], verbose=0)
        ypred=argmax(ypred)
        word=word2id(ypred, tokenizer)
        if word is None:
            break
        in_text+=' '+word
        if word == 'endseq':
            break
    return in_text


# In[45]:


tokenizer=load(open('tokenizer.pkl', 'rb'))
max_length=34
model=load_model('model_9.h5')
pic='test13.jpg'
photo=extract_features(pic)
description=generate_desc(model, tokenizer, photo, max_length)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
im = np.array(Image.open(pic))
plt.imshow(im)
print(description)


# In[ ]:




