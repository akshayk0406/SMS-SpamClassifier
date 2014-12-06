__author__ = 'akshaykulkarni'

import pandas
import csv
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline

def split_into_words(message):
    message = unicode(message,'utf8')
    return TextBlob(message).words

def split_into_lemma(message):
    words   = split_into_words(message)
    return [word.lemma for word in words]

'''
Part 1:-
Read the input file which contains the training data
columns in file are tab-seperated. Use Pandas library to read the file
'''

messages            = []
input_file_name     = './data/SMSSpamCollection'
messages            = pandas.read_csv(input_file_name,sep='\t',quoting=csv.QUOTE_NONE,names=['label','message'])
messages['length']  = messages['message'].map(lambda line:len(line))
train_sample_size   = int(0.70 * len(messages))

'''
Part 2:-
Data Preprocessing
- First split the message into words(tokens)
- Next extract base form of each word(normalize the text). eg "goes" and "go" carry same kind of information
'''

'''
Part 3:-
Convert each message, represented as a list of tokens, into a vector that machine learning models can understand.
We are using bag of words approach which involves:-
1) term frequency :- Number of times does a word occur in each message
2) Inverse document frequency:- Number of messages in which the word occurs
3) normalizing the vectors to unit length, to abstract from the original text length
'''

bag_of_words_model      = CountVectorizer(analyzer=split_into_lemma).fit(messages['message'])
messages_bow            = bag_of_words_model.transform(messages['message'])

tfidf_transformer       = TfidfTransformer().fit(messages_bow[:train_sample_size])
messages_tfidf          = tfidf_transformer.transform(messages_bow[:train_sample_size])
test_tfidf_transformer  = TfidfTransformer().fit(messages_bow[train_sample_size:])
test_messages_tfidf     = test_tfidf_transformer.transform(messages_bow[train_sample_size:])

label_train             = messages[:train_sample_size]['label']
label_test              = messages[train_sample_size:]['label']

'''
Part 4:-
Now messgaes are being represented as vectors and we can train our classifier
We will start with Naive Bayes Classifier
'''

Spam_ClassifierNB   = MultinomialNB().fit(messages_tfidf,label_train)
all_predictions     = Spam_ClassifierNB.predict(test_messages_tfidf)
print 'accuracy', accuracy_score(label_test, all_predictions)
print 'confusion matrix\n', confusion_matrix(label_test, all_predictions)

'''
Part 5:-
Using State Vector Machines for classifiction
'''

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemma)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

svm_detector    = grid_svm.fit(messages[:train_sample_size]['message'], label_train)
all_predictions = svm_detector.predict(messages[train_sample_size:]['message'])
print 'accuracy', accuracy_score(label_test, all_predictions)





