"""
Quora SW engineer Coding Challenge
Duplicate
Chin-Ting Ko  05/03/2017

This program is to  use machine learning algorithm to predict duplicate questions.
"""
import json
from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import sys

train_data = []
json_data = []
input_train_data = []
input_test_data = []
test_pair = []
expected_result = []
jason_pool ={}


def read_train_file():
    #with open("duplicate_sample.in", "r") as f:
        #all_lines = f.readlines()
        all_lines = sys.stdin.readlines()

        first_line = all_lines[0]
        if first_line.strip().isdigit():
            number_json = int(first_line.strip())

        json_lines = all_lines[1:number_json+1]
        for json_line in json_lines:
            json_object = json.loads(json_line)
            json_data.append(json_object)
            jason_pool[json_object['question_key']] = json_object['question_text']
        #print jason_pool

        next_line = all_lines[number_json+1]
        if next_line.strip().isdigit():
            number_pair = int(next_line.strip())

        duplicate_lines = all_lines[number_json+2:number_json+number_pair+2]
        for duplicate_line in duplicate_lines[:15000]:
            train_data.append(duplicate_line.split())

        test_number_line = all_lines[number_json+number_pair+2]
        if test_number_line.strip().isdigit():
            number_test = int(test_number_line.strip())

        test_lines = all_lines[number_json+number_pair+3:number_json+number_pair+number_test+3]
        for test_line in test_lines:
            test_pair.append((test_line.split()[0], test_line.split()[1]))
    #f.close()

#stop_words = set(stopwords.words('english'))
#stop_words.update(['.',  ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
#print stop_words


def process_train_data():
    for raw_data in train_data:
        first_question = ""
        second_question = ""
        temp = ""
        for x in jason_pool[raw_data[0]].split():
            #if x.lower().strip('?') not in stop_words:
            first_question += SnowballStemmer('english').stem(x.strip('?')) + " "
        for x in jason_pool[raw_data[1]].split():
            #if x.lower().strip('?') not in stop_words:
            second_question += SnowballStemmer('english').stem(x.strip('?')) + " "
        for x in first_question.split():
            if x not in second_question.split():
                temp += x + " "
        for x in second_question.split():
            if x not in first_question.split():
                temp += x + " "
        input_train_data.append(temp)
        expected_result.append(raw_data[2])
        #print input_train_data


def process_test_data():
    for raw_data in test_pair:
        first_question = ""
        second_question = ""
        temp = ""
        for x in jason_pool[raw_data[0]].split():
            #if x.lower().strip('?') not in stop_words:
            first_question += SnowballStemmer('english').stem(x.strip('?')) + " "
                #print first_question
        for x in jason_pool[raw_data[1]].split():
            #if x.lower().strip('?') not in stop_words:
            second_question += SnowballStemmer('english').stem(x.strip('?')) + " "
        for x in first_question.split():
            if x not in second_question.split():
                temp += x + " "
        for x in second_question.split():
            if x not in first_question.split():
                temp += x + " "
        input_test_data.append(temp)
    #print input_test_data


def predict(X, Y, test):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)
    tf_transformer_X = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer_X.transform(X_train_counts)
    tfidf_transformer_X = TfidfTransformer()
    X_train_tfidf = tfidf_transformer_X.fit_transform(X_train_counts)

    #n = len(X)
    #print n
    #X_train, X_test = X_train_tfidf[:int(n * 0.5)], X_train_tfidf[int(n * 0.5):]
    #Y_train, Y_test = Y[:int(n * 0.5)], Y[int(n * 0.5):]

    #model1 = tree.DecisionTreeClassifier()
    #model1 = BernoulliNB()
    #model = BernoulliNB()
    model = svm.SVC(kernel='rbf', gamma=0.7)
    #model1 = svm.SVC(kernel='rbf', gamma=0.7)
    #model1 = MultinomialNB()
    #model1 = svm.SVC(kernel='rbf', gamma=0.7)
    #model.fit(X,Y)

    #model1.fit(X_train, Y_train)
    model.fit(X_train_tfidf, Y)

    #print "Classifier trained."
    #print ""

    #expected = Y_test
    #predicted = model1.predict(X_test)
    #print(metrics.classification_report(expected, predicted))
    #print(metrics.confusion_matrix(expected, predicted))
    #return predicted.tolist()

    test_train_counts = count_vect.transform(test)
    tf_transformer_test = TfidfTransformer(use_idf=False).fit(test_train_counts)
    test_train_tf = tf_transformer_test.transform(test_train_counts)
    tfidf_transformer_test = TfidfTransformer()
    test_train_tfidf = tfidf_transformer_test.fit_transform(test_train_counts)

    return model.predict(test_train_tfidf).tolist()


def main():
    read_train_file()
    process_train_data()
    process_test_data()
    predict_result = list(predict(input_train_data, expected_result, input_test_data))
    #with open("output.txt", "w") as result:
    for x, y in zip(test_pair, predict_result):
        output = str(x).strip('()').replace(',', '').replace('\'', '') + ' '+str(y)
            #result.write(output+'\n')
        print output
    #print predict_result
    #write_file()
    #print zip(input_train_data, expected_result)


if __name__ == '__main__':
    main()