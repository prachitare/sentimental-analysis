import pandas
data= pandas.read_csv('data/train.tsv',sep='\t')
print(data.head())
print(data.Sentiment.value_counts())
import matplotlib.pyplot as plt
Sentiment_count= data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('reviews')
plt.ylabel('no. of reviews')
plt.show()
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv= CountVectorizer(lowercase=True,stop_words='english',ngram_range=(1,1),tokenizer=token.tokenize)
text_counts=cv.fit_transform(data['Phrase'])
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(text_counts,data['Sentiment'],test_size=.20,random_state=1)
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
clf=MultinomialNB().fit(xtrain,ytrain)
predicted=clf.predict(xtest)
print(metrics.accuracy_score(ytest,predicted))