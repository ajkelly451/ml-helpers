#!/usr/bin/env python

import nltk
import pandas as pd
import numpy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K


class VectorizeTextData(object):
    # TODO: ADD DOCSTRING
    def __init__(self, text_data_url, lemmatizer, pos_label=None, holdout_set_size=0.2, ngram_range=(1,1)):
        #super().__init__()

        # Read in text data from url. There should be only 2 columns, the first with the class labels and the second
        # with the associated text.
        text_data = pd.read_csv(text_data_url, sep='\t')
        text_data.columns = ['label', 'body_text']

        # Some qc checks on the inputs
        if len(text_data.columns) != 2:
            raise ValueError('"label" and "body_text" columns must be included in your supplied dataframe.')
        if lemmatizer not in ['lemmatize', 'stem']:
            raise ValueError('You must set lemmatizer to "lemmatize" or "stem"')
        if not holdout_set_size > 0 and not holdout_set_size < 1:
            raise ValueError('You must set a proportion for holdout_set_size strictly greater than 0 and less than 1.')

        # Set initial class attributes
        self.text_data = text_data
        self.holdout_set_size = holdout_set_size
        self.ngram_range = ngram_range
        self.pos_label = pos_label
        self.y_labels = ['label']
        if lemmatizer == 'lemmatize':
            self.lemmatizer = nltk.WordNetLemmatizer().lemmatize
        else:
            self.lemmatizer = nltk.PorterStemmer().stem

    @staticmethod
    def percent_is_punct(text):
        count = len([char for char in text if char in string.punctuation])
        return round(count/(len(text) - text.count(" ")), 3)

    @staticmethod
    def percent_is_cap(text):
        count = len(''.join(re.findall('[A-Z]+', text)))
        return round(count/(len(text) - text.count(" ")), 3)

    @ staticmethod
    def percent_is_num(text):
        count = len(''.join(re.findall('[0-9]+', text)))
        return round(count/(len(text) - text.count(" ")), 3)

    def clean_text(self, text):
        stopwords = nltk.corpus.stopwords.words('english')
        punct_and_num = string.punctuation + ''.join(str(i) for i in list(range(10)))
        text = "".join([char.lower() for char in text if char not in punct_and_num])
        tokens = re.split('\W+', text)
        text = [self.lemmatizer(word) for word in tokens if word not in stopwords]
        return text

    def engineer_features(self):
        eng_data = self.text_data
        eng_data['body_len'] = eng_data['body_text'].apply(lambda x: len(x) - x.count(" "))
        eng_data['body_len'] = eng_data['body_len']/max(eng_data['body_len'])  # turn it into percentage from 0 - 1
        eng_data['percent_punct'] = eng_data['body_text'].apply(lambda x: self.percent_is_punct(x))
        eng_data['percent_cap'] = eng_data['body_text'].apply(lambda x: self.percent_is_cap(x))
        eng_data['percent_num'] = eng_data['body_text'].apply(lambda x: self.percent_is_num(x))
        return eng_data

    def get_vectorized_data(self):
        # Engineer features
        eng_data = self.engineer_features()
        if eng_data['label'].nunique() > 2:
            eng_data = pd.get_dummies(eng_data, ['label'])
            self.y_labels = [col for col in eng_data.columns if 'label' in col]
        else:
            eng_data['label'] = [int(val == self.pos_label) for val in eng_data['label']]

        # get train/test split data for x and y
        x_train, x_test, y_train, y_test = train_test_split(
            eng_data[['body_text', 'body_len', 'percent_punct', 'percent_cap', 'percent_num']],
            eng_data[self.y_labels],
            test_size=self.holdout_set_size
        )

        # Fit vectorizer to training data
        tfidf_vect = TfidfVectorizer(analyzer=self.clean_text)
        tfidf_vect_fit = tfidf_vect.fit(x_train['body_text'])

        # Transform training data and test data with previously fit vectorizer
        tfidf_train = tfidf_vect_fit.transform(x_train['body_text'])
        tfidf_test = tfidf_vect_fit.transform(x_test['body_text'])

        # Concatenate tfidf vectors and engineered features
        x_train_vect = pd.concat([
            x_train[['body_len', 'percent_punct', 'percent_cap', 'percent_num']].reset_index(drop=True),
            pd.DataFrame(tfidf_train.toarray())], axis=1)
        x_test_vect = pd.concat([
            x_test[['body_len', 'percent_punct', 'percent_cap', 'percent_num']].reset_index(drop=True),
            pd.DataFrame(tfidf_test.toarray())], axis=1)

        return x_train_vect, x_test_vect, y_train, y_test


class TextClassificationNeuralNet(VectorizeTextData):
    # TODO: ADD DOCSTRING
    def __init__(self, number_hidden_layers, number_nodes_list, hidden_layer_activation, output_layer_activation,
                 optimizer, epochs, batch_size, **kwargs):
        super(TextClassificationNeuralNet, self).__init__(**kwargs)
        # TODO: Do verifications on input variables

        # Placeholders for test data and model
        self.x_test = None
        self.y_test = None
        self.model = None

        # Initialize class attributes
        self.number_hidden_layers = number_hidden_layers
        self.number_nodes_list = number_nodes_list
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    @staticmethod
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def fit_model(self):
        # Get vectorized and split data
        x_train, self.x_test, y_train, self.y_test = self.get_vectorized_data()

        # Build the Neural Network
        model = Sequential()
        model.add(Dense(units=self.number_nodes_list[0],
                        activation=self.hidden_layer_activation,
                        input_dim=len(x_train.columns)))
        for i in range(1, self.number_hidden_layers):
            model.add(Dense(units=self.number_nodes_list[i],
                            activation=self.hidden_layer_activation))
        model.add(Dense(1, activation=self.output_layer_activation))

        # Compile Neural Network
        if len(self.y_labels) == 1:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        model.compile(loss=loss, optimizer=self.optimizer, metrics=['accuracy', self.precision_m, self.recall_m])
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        self.model = model

    def get_performance_metrics(self):
        # Make sure model was fit
        if not self.model:
            raise Exception('You must first fit the model to obtain metrics.')

        # Test final performance metrics on holdout data
        # TODO: HANDLE MULTICLASS BETTER
        _, accuracy, precision, recall = self.model.evaluate(self.x_test, self.y_test)
        # if len(self.y_test.columns) == 1:
        #     predictions = self.model.predict(self.x_test)
        #     contingency_table = pd.crosstab(self.y_test['label'].to_numpy(), numpy.array(predictions))
        # else:
        #     contingency_table = None
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall}  #, 'contingency_table': contingency_table}


if __name__ == "__main__":

    text_nn = TextClassificationNeuralNet(
        text_data_url='SMSSpamCollection.tsv',
        lemmatizer='lemmatize',
        pos_label='spam',
        holdout_set_size=0.2,
        ngram_range=(1, 2),
        number_hidden_layers=2,
        number_nodes_list=[100, 50],
        hidden_layer_activation='relu',
        output_layer_activation='sigmoid',
        optimizer='sgd',
        epochs=20,
        batch_size=100
    )
    text_nn.fit_model()
    print(text_nn.get_performance_metrics())