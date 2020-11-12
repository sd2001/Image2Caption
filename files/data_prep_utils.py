import modules as m
import data_visualize as v

def make_txt(file):
    f=open(file,'r')
    text=f.read()
    f.close()
    return text

def form_dict(img_list):
    img_dict={}
    for im in img_list:
        if im in v.ic:
            img_dict[im]=v.ic[im]
    return img_dict

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

import string
def clean_dict(des):
    table=str.maketrans('', '', string.punctuation)
    for k,desc_list in des.items():
        for i in range(len(desc_list)):
            desc=desc_list[i]
            desc=desc.split()
            desc=[word.lower() for word in desc]
            desc=[w.translate(table) for w in desc]
            desc=[word for word in desc if len(word)>1]
            desc=[word for word in desc if word.isalpha()]
            desc_list[i]= ' '.join(desc)

def vocab_create(des):
    all1=set()
    for k in des.keys():
        [all1.update(d.split()) for d in des[k]]
    return all1

def save_descriptions(descriptions):
    lines=[]
    for key,d in descriptions.items():
        for desc in d:
            lines.append(key + ' ' + desc)
    data='\n'.join(lines)
    file=open('descriptions.txt', 'w')
    file.write(data)
    file.close()

def make_set(file):
    txt=make_txt(file)
    d=[]
    for line in txt.split('\n'):
        if len(line)<1:
            continue
        sent=line.split('.')[0]
        d.append(sent)
    return set(d)

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

def load_photos(file,dataset):
    all_features=m.load(open(file, 'rb'))
    f={k: all_features[k] for k in dataset}
    return f

def max_length(des):
    lines=to_lines(des)
    return max(len(d.split()) for d in lines)

def to_lines(des):
    all=[]
    for key in des.keys():
        [all.append(d) for d in des[key]]
    return all

def create_tokenizer(des):
    lines=to_lines(des)
    tokenizer=m.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
    X1,X2,y=[],[],[]
    for desc in desc_list:
        seq=tokenizer.texts_to_sequences([desc])[0]
        for i in range(1,len(seq)):
            in_seq,out_seq=seq[:i], seq[i]
            in_seq=m.pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq=m.to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return m.array(X1), m.array(X2), m.array(y)

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
    while 1:
        for key,desc_list in descriptions.items():
            photo=photos[key][0]
            in_img,in_seq, out_word=create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
            yield ([in_img,in_seq],out_word)