import re
import urllib
import json
import numpy as np
import pandas as pd
import csv
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
import random

stopwords = stopwords.words("english")

class Classifier():
    def __init__(self):
        self.classifier = None
        self.df = None

        with open('config.json') as config_file:
            data = json.load(config_file)
        
        self.wassa = data["WASSA"]
        self.archive = data["archive"]
        self.corporate = data["corporate"]

        self.global_dataset = np.array([])
        self.global_processed_model = {}

        self.init_wassa()
        self.init_archive()
        self.init_corporate()
        self.train_model()

    
    def clearify_wassa_post(self, post):
        data = list(map(lambda x: x.rstrip(), post.split('\t')))
        return { "text": data[1], "emotion": data[2] }

    def get_wassa_dataset(self, url):
        file = urllib.request.urlopen(url)
        posts = list(map(lambda x : x.decode("utf-8"), file.readlines()))
        return list(map(self.clearify_wassa_post, posts))

    def init_wassa(self):
        for key in self.wassa:
            for dataset_version in self.wassa[key]:
                local_dataset = self.get_wassa_dataset(self.wassa[key][dataset_version])
                self.global_dataset = np.concatenate((self.global_dataset, np.array(local_dataset)))
            
    def clearify_archive_comment(self, comment):
        clear_comment = comment.rstrip().split(';')
        return { "text" : clear_comment[0], "emotion": clear_comment[1] }

    def get_archive_dataset(self, url):
        with open(url) as file:
            data = file.readlines()
        return list(map(self.clearify_archive_comment, data))
    
    def init_archive(self):
        for key in self.archive:
            for dataset_version in self.archive[key]:
                local_dataset = self.get_archive_dataset(self.archive[key][dataset_version])
                self.global_dataset = np.concatenate((self.global_dataset, np.array(local_dataset)))
    
    def clearify_corporate(self, text):
        return { "text" : text, "emotion" : "corporate" }

    def init_corporate(self):
        with open(self.corporate, encoding = "ISO-8859-1") as csvfile:
            corporate_reader = csv.DictReader(csvfile, delimiter=',')
            reviews = [row['text'] for row in corporate_reader]
            local_dataset = list(map(self.clearify_corporate, reviews))
            self.global_dataset = np.concatenate((self.global_dataset, np.array(local_dataset)))
    
    def remove_noise(self, tokens, stop_words = ()):
        stemmer = SnowballStemmer("english")
        cleaned_tokens = []
        for token in tokens:
            if len(token) > 0 and not re.search(r'[^0-9a-zA-Z]+', token) and token.lower() not in stop_words:
                cleaned_tokens.append(stemmer.stem(token))
        return cleaned_tokens

    def get_all_words(self, cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token

    def get_tokens_for_model(self, cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tokens)
    
    def process_model(self):
        self.df = pd.DataFrame(list(self.global_dataset))

        emotions = self.df['emotion'].drop_duplicates().tolist()

        for em in emotions:
            dataset = self.df[self.df['emotion'] == em]['text'].astype('str').to_numpy()
            
            tokens = [nltk.word_tokenize(text) for text in dataset]
            cleaned_tokens = [self.remove_noise(token, stopwords) for token in tokens]
            
            self.global_processed_model[em] = [(text_dict, em) for text_dict in self.get_tokens_for_model(cleaned_tokens)]

    def train_model(self):
        self.process_model()

        dataset_for_model = []
        for key in self.global_processed_model.keys():
            dataset_for_model += self.global_processed_model[key]
        
        # random.shuffle(dataset_for_model)

        # text_count = (self.df.shape[0] * 80) // 100

        # train_data = dataset_for_model[:text_count]
        # test_data = dataset_for_model[text_count:]

        # print("train data:", len(train_data))
        # print("test data:", len(test_data))

        self.classifier = NaiveBayesClassifier.train(dataset_for_model)

        # print("Accuracy is:", classify.accuracy(self.classifier, test_data))
        # print(self.classifier.show_most_informative_features(10))
    
    def classify_message(self, text):
        tokens = self.remove_noise(nltk.tokenize.word_tokenize(text))
        return self.classifier.classify(dict([token, True] for token in tokens))