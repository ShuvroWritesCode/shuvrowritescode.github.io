---
layout: post
title: Data Preprocessing in Generative AI
dates: 14-01-2025
categories: [Generative AI]
tag: [Generative AI, Data Preprocessing, Machine Learning, Natural Language Processing, Text Cleaning, NLP]
description:  
image:
    path: /assets/img/headers/post2.webp
    lqip: data:image/webp;base64,UklGRoIAAABXRUJQVlA4IHYAAACQAwCdASoUAAoAPpE4l0eloyIhMAgAsBIJZQAAW9leZAzPGi/gAP7XP869nDh+xTmrig7IkwQJYz4nl3S1T9f/OOI3nmdD/+pTlehBHqCwleUGnxrGML1cX3/b3wtuHSqwe3iV4eY3w9RqbYqoaiqkWzyC+gAA
---

Data preprocessing is a **critical step** in generative AI pipelines. It ensures the input data is clean, consistent, and optimized for model training, directly impacting the **model’s performance**. Here, in this blog we will learn about different data preprocessing techniques. 

[View Notebook](/assets/ipynb-files/Text_Preprocessing.ipynb)

I used the famous *IMDB kaggle dataset*. Database link: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

```python
import pandas as pd
```

```python
!pwd
```
> */content*

```python
data_path = "/content/IMDB Dataset.csv"
df = pd.read_csv(data_path)
df.shape
```
> (50000, 2)

```python
df.head()
```

![alt](</assets/img/2025-01-15-post-2/Screenshot 2025-01-15 at 3.33.28 AM.png>)


```python
df = df.head(100)
df.shape
```
> (100, 2)

```python
df.head()
```

![alt](</assets/img/2025-01-15-post-2/Screenshot 2025-01-15 at 3.36.09 AM.png>)

# Data Preprocessing Walkthrough
---
## Lower Case Operation
Text normalization starts with converting all text to lowercase to ensure uniformity.

```python
df['review'][3]
```

> Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />This movie is slower than a soap opera... and suddenly, Jake decides to become Rambo and kill the zombie.<br /><br />OK, first of all when you're going to make a film you must Decide if its a thriller or a drama! As a drama the movie is watchable. Parents are divorcing & arguing like in real life. And then we have Jake with his closet which totally ruins all the film! I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. As for the shots with Jake: just ignore them.

```python
df['review'] = df['review'].str.lower()
df['review'][3]
```

Transformed text in lowercase.

> basically there's a family where a little boy (jake) thinks there's a zombie in his closet & his parents are fighting all the time.<br /><br />this movie is slower than a soap opera... and suddenly, jake decides to become rambo and kill the zombie.<br /><br />ok, first of all when you're going to make a film you must decide if its a thriller or a drama! as a drama the movie is watchable. parents are divorcing & arguing like in real life. and then we have jake with his closet which totally ruins all the film! i expected to see a boogeyman similar movie, and instead i watched a drama with some meaningless thriller spots.<br /><br />3 out of 10 just for the well playing parents & descent dialogs. as for the shots with jake: just ignore them.

```python
df
```

![alt](</assets/img/2025-01-15-post-2/Screenshot 2025-01-15 at 3.39.39 AM.png>)

---

## Remove URL

URLs in the text are irrelevant and should be removed using regex.

```python
# prompt: write a python code using regex to remove all URL from the dataset.

def remove_urls(text):
	url_pattern = re.compile(r'https?://\S+|www\.\S+')
	return url_pattern.sub(r'', text)

df['review'] = df['review'].apply(remove_urls)

df
```

![alt](</assets/img/2025-01-15-post-2/Screenshot 2025-01-15 at 3.43.39 AM.png>)

---

## Punctuation Handling

Punctuation can often be removed to simplify text analysis.

```python
import string
exclude = string.punctuation
def remove_punc(text):
	return text.translate(str.maketrans('', '', exclude))
df['review'][5]
```
> probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it's not preachy or boring. it just never gets old, despite my having seen it some 15 or more times in the last 25 years. paul lukas' performance brings tears to my eyes, and bette davis, in one of her very few truly sympathetic roles, is a delight. the kids are, as grandma says, more like "dressed-up midgets" than children, but that only makes them more fun to watch. and the mother's slow awakening to what's happening in the world and under her own roof is believable and startling. if i had a dozen thumbs, they'd all be "up" for this movie.

```python
remove_punc(df['review'][5])
```
>probably my alltime favorite movie a story of selflessness sacrifice and dedication to a noble cause but its not preachy or boring it just never gets old despite my having seen it some 15 or more times in the last 25 years paul lukas performance brings tears to my eyes and bette davis in one of her very few truly sympathetic roles is a delight the kids are as grandma says more like dressedup midgets than children but that only makes them more fun to watch and the mothers slow awakening to whats happening in the world and under her own roof is believable and startling if i had a dozen thumbs theyd all be up for this movie

---

## Chat keywords handling

Here, we try to handle different shortcuts and keywords that we use in our day to day chats on social media.

```python
chat_words = {

	'AFAIK':'As Far As I Know',
	'AFK':'Away From Keyboard',
	'ASAP':'As Soon As Possible',
	"FYI": "For Your Information",
	"ASAP": "As Soon As Possible",
	"BRB": "Be Right Back",
	"BTW": "By The Way",
	"OMG": "Oh My God",	
	"IMO": "In My Opinion",
	"LOL": "Laugh Out Loud",
	"TTYL": "Talk To You Later",
	"GTG": "Got To Go",
	"TTYT": "Talk To You Tomorrow",
	"IDK": "I Don't Know",
	"TMI": "Too Much Information",
	"IMHO": "In My Humble Opinion",
	"ICYMI": "In Case You Missed It",
	"AFAIK": "As Far As I Know",
	"BTW": "By The Way",
	"FAQ": "Frequently Asked Questions",
	"TGIF": "Thank God It's Friday",
	"FYA": "For Your Action",
	"ICYMI": "In Case You Missed It",
}
```

```python
def chat_words_conversion(text):
	new_text = []
	for w in text.split():
		if w.upper() in chat_words:
			new_text.append(chat_words[w.upper()])
		else:
			new_text.append(w)
return " ".join(new_text)

chat_words_conversion('Do this work ASAP')
```

> Do this work As Soon As Possible

---

## Spelling Correction

Handling the spelling mistakes and mistyped words. Handle spelling mistakes using tools like `TextBlob`.

```python
from textblob import TextBlob

def spelling_correction(text):
	blob = TextBlob(text)
	return blob.correct()

df['review'] = df['review'].apply(spelling_correction)
```

---

## Remove Stopwords

Stopwords (e.g., “the”, “is”) add noise and can be removed.

```python
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
```
![alt](</assets/img/2025-01-15-post-2/Screenshot 2025-01-15 at 3.49.11 AM.png>)

```python
stopwords.words('english')
```

```
['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 "don't",
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'ain',
 'aren',
 "aren't",
 'couldn',
 "couldn't",
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'isn',
 "isn't",
 'ma',
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'shouldn',
 "shouldn't",
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'won',
 "won't",
 'wouldn',
 "wouldn't"]
```

```python
def remove_stopwords(text):
	stop_words = stopwords.words('english')
	words = text.split()
	new_text = [w for w in words if w not in stop_words]
	return " ".join(new_text)

df['review'] = df['review'].apply(remove_stopwords)

df
```

![alt](</assets/img/2025-01-15-post-2/Screenshot 2025-01-15 at 3.51.26 AM.png>)

---

## Removing the Emoji(s)
Clean text by removing emojis and special symbols.

```python
import re

def remove_emojis(text):
	emoji_pattern = re.compile("["
						u"\U0001F600-\U0001F64F" # emoticons
						u"\U0001F300-\U0001F5FF" # symbols & pictographs
						u"\U0001F680-\U0001F6FF" # transport & map symbols
						u"\U0001F1E0-\U0001F1FF" # flags (iOS)
						u"\U00002702-\U000027B0"
						u"\U000024C2-\U0001F251"
						"]+", flags=re.UNICODE)

	return emoji_pattern.sub(r'', text)

df['review'] = df['review'].apply(remove_emojis)

df
```

![[Screenshot 2025-01-15 at 3.53.26 AM.png]]

---

## Tokenization

Tokenize text into words or sentences for further processing.

### Using the Split Function

```python
# Words tokenization
sent1 = "I am going to Delhi"
sent1.split()
```
> ['I', 'am', 'going', 'to', 'Delhi']

```python
# Sentence tokenization
sent2 = "I am going to Europe. I am going to stay there for a month. I will be back soon."
sent2.split('.')
```
> ['I am going to Europe', ' I am going to stay there for a month', ' I will be back soon', '']

There are many other tokenizing techniques using nltk, regex, etc.

## Stemmer

Stemmer is basically digging out the root words out of the data for better optimized calculations later.

```python
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_words(text):
	words = text.split()
	stemmed_words = [stemmer.stem(word) for word in words]
	return " ".join(stemmed_words)
```

```python
sample = "walk walks walking walked"
stem_words(sample)
```
> walk walk walk walk

```python
df['review'] = df['review'].apply(stem_words)
```
---
## Lemmatization

A more context-aware process than stemming.

```python
import nltk

from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('wordnet')

nltk.download('omw-1.4')

wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('punkt_tab') # Download the necessary tokenizer

  

sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."

punctuations="?:!.,;"

sentence_words = nltk.word_tokenize(sentence)

for word in sentence_words:

if word in punctuations:

sentence_words.remove(word)

  

sentence_words

print("{0:20}{1:20}".format("Word","Lemma"))

for word in sentence_words:

print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word,pos='v')))
```

![alt](</assets/img/2025-01-15-post-2/Screenshot 2025-01-15 at 3.59.24 AM.png>)

> NOTE: Stemming & lamatization are same to retrieve root words but lamatization is worked good. Lamatization is slow & stemming is fast

--- 


