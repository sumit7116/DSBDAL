# Generally the data is textual
# 1) High dimensionality (number of unique words)
# 2) Unstructured


# Applications from textual data
# 1) Sentiments behind the data (Yes / No, Positive / Negative )
# 2) We can predict some outcomes (polls)

# What are the steps for processing textual data?
# 1) Gather the data (social media, news, books, reviews)
# 2) Divide the data into words
# 3) Pre-processing (stop words removal, punctuations removal)
# 4) POS Tagging
# 5) Stemming
# 6) Lemmatization : It is same as stemming only difference is that it uses vocabulary

# 7) Find out Term Frequency (TF)
# 8) Find out Inverse Document Frequency (IDF)

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

# nltk.download('stopwords')
# from nltk.corpus import stopwords
# 
# stop_words = set(stopwords.words('english'))

txt = "Sukanya, Rajib and Naba are my good frind. " \
    "Sukanya is getting married next year. " \
    "Marriage is a big step in one’s life." \
    "It is both exciting and frightening. " \
    "But friendship is a sacred bond between people." \
    "It is a special kind of love between us. " \
    "Many of you must have tried searching for a friend "\
    "but never found the right one."

#Convert/divide the text into lines
tokenized = sent_tokenize(txt)
print(tokenized)

# Convert the text into words
tokens = nltk.word_tokenize(txt)   
print(tokens)

nltk.download('averaged_perceptron_tagger')
tagged_words = nltk.pos_tag(tokens)
print(tagged_words)

from nltk.stem import PorterStemmer
p = PorterStemmer()

from nltk.stem import LancasterStemmer
l = LancasterStemmer()

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemt = WordNetLemmatizer()


print('\nword \t Stemmed Word')
print('\n-----------------------------')
for word in tokens:
  pstemmedword = p.stem(word)
  lstemmedword = l.stem(word)
  lematizeword = lemt.lemmatize(word)
  print(word, '\t', pstemmedword, '\t', lstemmedword, '\t', lematizeword)