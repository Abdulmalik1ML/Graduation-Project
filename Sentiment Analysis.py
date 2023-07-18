#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# sns.set
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
stemmer = nltk.SnowballStemmer('english')
nltk.download('vader_lexicon')


# In[2]:


data = pd.read_csv("Tiktok_reviews.csv")
data=data.dropna()
data


# In[3]:


data.info()


# In[4]:


data['liked'].value_counts()


# In[5]:


sns.countplot(x=data['liked'])


# In[6]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

text = " ".join(i for i in data.review)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[7]:


data['review'][3]


# In[8]:


from nltk.corpus import stopwords


# In[9]:


nltk.download('stopwords')


# In[10]:


stopword=set(stopwords.words('english'))
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data['review']=data['review'].apply(clean)


# In[11]:


data=data.dropna()
data


# In[12]:


from nltk.stem import PorterStemmer


# In[13]:


ps = PorterStemmer()


# In[14]:


s=data['review'][3]
s


# In[15]:


import sklearn
from sklearn.feature_extraction.text import CountVectorizer


# In[16]:


cv=CountVectorizer()


# In[17]:


cv.fit_transform(s.split()).toarray()


# In[18]:


corpus=[]

for i in range(len(data)):
    s=re.sub('[^a-zA-Z]', " ",data['review'][i])
    s=s.lower()
    s=s.split()
    s= [word for word in s if word not in stopwords.words('english')]
    s= ' '.join(s)
    s= ps.stem(s)
    corpus.append(s)
corpus


# In[19]:


cv.fit_transform(corpus).toarray()


# In[20]:


cv.fit_transform(corpus).toarray().shape


# In[21]:


x= cv.fit_transform(corpus).toarray()
x


# In[22]:


y=data['liked']
y


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=42)
x_train


# In[25]:


x_train.shape


# In[26]:


x_test


# In[27]:


x_test.shape


# In[28]:


from sklearn.naive_bayes import MultinomialNB


# In[29]:


clf = MultinomialNB()
clf.fit(x_train, y_train)


# In[30]:


y_pred = clf.predict(x_test)
y_pred


# In[31]:


y_test.values


# In[32]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[33]:


print(confusion_matrix(y_test, y_pred))


# In[34]:


print(accuracy_score(y_test, y_pred))


# In[35]:


print(classification_report(y_test, y_pred))


# In[36]:


confusion_matrix = np.array([[85, 103], [47, 765]])
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[37]:


# model 
from sklearn.tree import DecisionTreeClassifier


# In[38]:


clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)


# In[39]:


y_pred = clf.predict(x_test)
y_pred


# In[40]:


y_test.values


# In[41]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# In[42]:


print(confusion_matrix(y_test, y_pred))


# In[43]:


print(accuracy_score(y_test, y_pred))


# In[44]:


print(classification_report(y_test, y_pred))


# In[45]:


confusion_matrix = np.array([[75, 113], [64, 748]])
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='g')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[ ]:




