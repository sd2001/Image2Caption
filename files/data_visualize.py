import modules as m

captions=open(m.IMG_CAPTION, 'r').read().split("\n")
x_train=open(m.TRAIN_IMG_NAME, 'r').read().split("\n")
x_val=open(m.VALID_IMG_NAME, 'r').read().split("\n")
x_test=open(m.TEST_IMG_NAME, 'r').read().split("\n")

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

combined_df=m.DataFrame(combined,columns=['Image','Caption'])
ds=combined_df.values
m.nltk.download('punkt')

final_corpus=[]
dup_corpus=[]
for sent in corpus:
    words=m.word_tokenize(sent)
    
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

fdist1=m.nltk.FreqDist(dup_corpus)
fd=fdist1.most_common()
words=[]
aa=[]
for i in range(len(fd)):
    aa=[]
    aa.append(fd[i][0])
    aa.append(fd[i][1])
    words.append(aa)

df=m.DataFrame(words,columns=['Words','Count'])
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