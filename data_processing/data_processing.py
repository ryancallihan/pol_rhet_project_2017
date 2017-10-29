import gensim
import nltk
import re
import numpy as np
import pandas as pd
import pickle as p
from sklearn.preprocessing import scale
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


class Data_Processing:
    def __init__(self, load_tokenizers=False):
        self.load_tokenizers = load_tokenizers
        self.x_tokenizer = Tokenizer()
        self.y_tokenizer = Tokenizer()
        if load_tokenizers:
            self.x_tokenizer = p.load(file=open("../../data_processing/x_tokenizer.p", "rb"))
            self.y_tokenizer = p.load(file=open("../../data_processing/y_tokenizer.p", "rb"))

    def load_tsv(self, filename):
        """
        Loads pandas dataframe from file
        :param filename:
        :return:
        """
        return pd.read_csv(filename, sep='\t', error_bad_lines=False)

    def load_dataframe(self, filename):
        """
        Returns pandas dataframe from pickle
        :param filename:
        :return:
        """
        print("FILENAME: ", filename)
        return p.load(open(filename, "rb"))

    def shuffle_dataframe(self, dataframe):
        """
        Shuffles rows of pandas dataframe
        :param dataframe:
        :return:
        """
        return dataframe.iloc[np.random.permutation(len(dataframe))]

    def retrieve_features(self, dataframe):
        """
        Retrieves features (X) from dataframe
        :param dataframe:
        :return:
        """
        return list(dataframe["tweet"])

    def retrieve_labels(self, dataframe):
        """
        Retrieves labels (Y) from pandas dataframe
        Returns years | months | partys
        Separates months into quarters
        :param dataframe:
        :return:
        """
        years = list([str(int(year)) for year in dataframe["year"]])
        months = list([int(month) for month in dataframe["month"]])
        partys = list(dataframe["affiliation"])
        for idx, month in enumerate(months):
            month = int(month)
            if 0 < month <= 4:
                months[idx] = str(1)
            elif 3 < month <= 6:
                months[idx] = str(2)
            elif 6 < month <= 9:
                months[idx] = str(3)
            elif 9 < month <= 12:
                months[idx] = str(4)
            else:
                months[idx] = str("XXX")

        return years, months, partys

    def vectorize_text(self, texts, is_label=False):
        """
        Takes a list of texts and returns a matrix of "bag of word" vectors.
        Each row represents a text with a vector of length: number of types.
        includes a 1 for each type in the text.
        :param texts:
        :param tokenizer: # Keras tokenizer object
        :return:
        """
        if is_label:
            tokenizer = self.y_tokenizer
        else:
            tokenizer = self.x_tokenizer
        indexed = tokenizer.texts_to_sequences(texts)
        num_items = len(tokenizer.word_index)
        vectors = []
        for index in indexed:
            index = index
            vec = np.zeros(num_items, dtype=np.int).tolist()
            for idx in index:
                vec[idx - 1] = 1
            vectors.append(vec)
        return vectors

    def create_embedding_layer(self, w2v_path, is_label=False):
        """
        Creates an embeddings layer for use with tensorflow
        Help received for this from Mai Mayeg
        :param w2v_path:
        :param is_label:
        :return:
        """
        if is_label:
            tokenizer = self.y_tokenizer
        else:
            tokenizer = self.x_tokenizer
        w2v_model = self.load_w2v_model(w2v_path)
        num_words = len(tokenizer.word_index) + 1
        w2v_vector_len = w2v_model.vector_size
        w2v_embeddings = np.zeros((num_words, w2v_vector_len))

        for word, index in tokenizer.word_index.items():
            try:
                w2v_embeddings[index] = w2v_model[word]
            except KeyError:
                continue
        return w2v_embeddings

    def load_w2v_model(self, w2v_path):
        """
        Loads w2v model
        :param w2v_path:
        :return:
        """
        return gensim.models.Word2Vec.load(w2v_path)

    def pad_sequences(self, sentences, max_len):
        """
        Pads sequences for uniform length
        :param sentences:
        :param max_len:
        :return:
        """
        padded_sentences = []
        for sent in sentences:
            padded_sent = []
            sent_len = len(sent)
            for idx in range(1, max_len):
                if idx <= sent_len:
                    padded_sent.append(sent[idx - 1])
                else:
                    padded_sent.append(0)
            padded_sentences.append(padded_sent)
        return padded_sentences

    def build_vectors(self, train, test, is_label=False):
        """
        Creates vectors for each tweet
        :param train:
        :param test:
        :param is_label:
        :return:
        """
        if is_label:
            tokenizer = self.y_tokenizer
        else:
            tokenizer = self.x_tokenizer
        train_vectorized = self.vectorize_text(train, tokenizer)
        test_vectorized = self.vectorize_text(test, tokenizer)
        return train_vectorized, test_vectorized  # , tokenizer

    def build_indices(self, train, test, maxlen=-1, pad=True, is_label=False):
        """
        Builds index vectors for each tweet
        :param train:
        :param test:
        :param maxlen:
        :param pad:
        :param is_label:
        :return:
        """

        if maxlen == -1:
            maxlen = len(
                max(train, key=len)
            )

        train = self.vectorize_text(train, is_label=is_label)
        test = self.vectorize_text(test, is_label=is_label)

        if pad:
            train = self.pad_sequences(train, max_len=maxlen)
            test = self.pad_sequences(test, max_len=maxlen)
        return train, test

    def subset(self, data_train, data_test, section):
        """
        Creates a subset of the data by quarter
        :param data_train:
        :param data_test:
        :param section:
        :return:
        """
        if section == 1:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2015.0) &
                ((data_train['month'] == 1.0) |
                 (data_train['month'] == 2.0) |
                 (data_train['month'] == 3.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2015.0)) &
                ((data_test['month'] == 1.0) |
                 (data_test['month'] == 2.0) |
                 (data_test['month'] == 3.0))
                ]
        elif section == 2:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2015.0) &
                ((data_train['month'] == 4.0) |
                 (data_train['month'] == 5.0) |
                 (data_train['month'] == 6.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2015.0)) &
                ((data_test['month'] == 4.0) |
                 (data_test['month'] == 5.0) |
                 (data_test['month'] == 6.0))
                ]
        elif section == 3:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2015.0) &
                ((data_train['month'] == 7.0) |
                 (data_train['month'] == 8.0) |
                 (data_train['month'] == 9.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2015.0)) &
                ((data_test['month'] == 7.0) |
                 (data_test['month'] == 8.0) |
                 (data_test['month'] == 9.0))
                ]
        elif section == 4:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2015.0) &
                ((data_train['month'] == 10.0) |
                 (data_train['month'] == 11.0) |
                 (data_train['month'] == 12.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2015.0)) &
                ((data_test['month'] == 10.0) |
                 (data_test['month'] == 11.0) |
                 (data_test['month'] == 12.0))
                ]
        elif section == 5:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2016.0) &
                ((data_train['month'] == 1.0) |
                 (data_train['month'] == 2.0) |
                 (data_train['month'] == 3.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2016.0)) &
                ((data_test['month'] == 1.0) |
                 (data_test['month'] == 2.0) |
                 (data_test['month'] == 3.0))
                ]
        elif section == 6:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2016.0) &
                ((data_train['month'] == 4.0) |
                 (data_train['month'] == 5.0) |
                 (data_train['month'] == 6.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2016.0)) &
                ((data_test['month'] == 4.0) |
                 (data_test['month'] == 5.0) |
                 (data_test['month'] == 6.0))
                ]
        elif section == 7:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2016.0) &
                ((data_train['month'] == 7.0) |
                 (data_train['month'] == 8.0) |
                 (data_train['month'] == 9.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2016.0)) &
                ((data_test['month'] == 7.0) |
                 (data_test['month'] == 8.0) |
                 (data_test['month'] == 9.0))
                ]
        elif section == 8:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2016.0) &
                ((data_train['month'] == 10.0) |
                 (data_train['month'] == 11.0) |
                 (data_train['month'] == 12.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2016.0)) &
                ((data_test['month'] == 10.0) |
                 (data_test['month'] == 11.0) |
                 (data_test['month'] == 12.0))
                ]
        elif section == 9:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2017.0) &
                ((data_train['month'] == 1.0) |
                 (data_train['month'] == 2.0) |
                 (data_train['month'] == 3.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2017.0)) &
                ((data_test['month'] == 1.0) |
                 (data_test['month'] == 2.0) |
                 (data_test['month'] == 3.0))
                ]

        elif section == 10:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                (data_train['year'] == 2017.0) &
                ((data_train['month'] == 4.0) |
                 (data_train['month'] == 5.0) |
                 (data_train['month'] == 6.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2017.0)) &
                ((data_test['month'] == 4.0) |
                 (data_test['month'] == 5.0) |
                 (data_test['month'] == 6.0))
                ]

        elif section == 11:
            data_train_R = data_train.loc[
                (data_train['affiliation'] == "R") &
                (data_train['year'] == 2017.0) &
                ((data_train['month'] == 1.0) |
                 (data_train['month'] == 2.0) |
                 (data_train['month'] == 3.0))
                ]

            data_train_D = data_train.loc[
                (data_train['affiliation'] == "D") &
                (data_train['year'] == 2017.0) &
                ((data_train['month'] == 1.0) |
                 (data_train['month'] == 2.0) |
                 (data_train['month'] == 3.0))
                ]

            data_test_R = data_test.loc[
                (data_test['affiliation'] == "R") &
                ((data_test['year'] == 2017.0)) &
                ((data_test['month'] == 1.0) |
                 (data_test['month'] == 2.0) |
                 (data_test['month'] == 3.0))
                ]

            data_test_D = data_test.loc[
                (data_test['affiliation'] == "D") &
                ((data_test['year'] == 2017.0)) &
                ((data_test['month'] == 1.0) |
                 (data_test['month'] == 2.0) |
                 (data_test['month'] == 3.0))
                ]

            data_test_D, _ = np.split(self.shuffle_dataframe(data_test_D),
                                      [int(.5 * len(data_test_D))])

            data_train_D, _ = np.split(self.shuffle_dataframe(data_train_D),
                                       [int(.5 * len(data_train_D))])

            print("SIZE OF D TRAIN:", len(data_train_D))
            print("SIZE OF D TEST:", len(data_test_D))
            print("SIZE OF R TRAIN:", len(data_train_R))
            print("SIZE OF R TEST:", len(data_test_R))

            data_train = data_train_R.append(data_train_D)
            data_test = data_test_R.append(data_test_D)


        elif section == 12:
            data_train_R = data_train.loc[
                (data_train['affiliation'] == "R") &
                (data_train['year'] == 2017.0) &
                ((data_train['month'] == 4.0) |
                 (data_train['month'] == 5.0) |
                 (data_train['month'] == 6.0))
                ]

            data_train_D = data_train.loc[
                (data_train['affiliation'] == "D") &
                (data_train['year'] == 2017.0) &
                ((data_train['month'] == 4.0) |
                 (data_train['month'] == 5.0) |
                 (data_train['month'] == 6.0))
                ]

            data_test_R = data_test.loc[
                (data_test['affiliation'] == "R") &
                ((data_test['year'] == 2017.0)) &
                ((data_test['month'] == 4.0) |
                 (data_test['month'] == 5.0) |
                 (data_test['month'] == 6.0))
                ]

            data_test_D = data_test.loc[
                (data_test['affiliation'] == "D") &
                ((data_test['year'] == 2017.0)) &
                ((data_test['month'] == 4.0) |
                 (data_test['month'] == 5.0) |
                 (data_test['month'] == 6.0))
                ]

            data_test_D, _ = np.split(self.shuffle_dataframe(data_test_D),
                                      [int(.5 * len(data_test_D))])

            data_train_D, _ = np.split(self.shuffle_dataframe(data_train_D),
                                       [int(.5 * len(data_train_D))])

            print("SIZE OF D TRAIN:", len(data_train_D))
            print("SIZE OF D TEST:", len(data_test_D))
            print("SIZE OF R TRAIN:", len(data_train_R))
            print("SIZE OF R TEST:", len(data_test_R))

            data_train = data_train_R.append(data_train_D)
            data_test = data_test_R.append(data_test_D)

        else:
            data_train = data_train.loc[
                ((data_train['affiliation'] == "R") |
                 (data_train['affiliation'] == "D")) &
                ((data_train['year'] == 2015.0) |
                 (data_train['year'] == 2016.0) |
                 (data_train['year'] == 2017.0))
                ]

            data_test = data_test.loc[
                ((data_test['affiliation'] == "R") |
                 (data_test['affiliation'] == "D")) &
                ((data_test['year'] == 2015.0) |
                 (data_test['year'] == 2016.0) |
                 (data_test['year'] == 2017.0))
                ]

        print("SECTION>>>>", section)
        return data_train, data_test

    def run(self, train_file, test_file, section=-1, x_mode=None, y_mode=None,
            feat_vec_len=-1, shuffle=True):
        """
        Returns all training and testing data
        (x_train, y_train, x_test, y_test, x_tokenizer, y_tokenizer)
        :param y_mode:
        :param x_mode:
        :param train_file:
        :param test_file:
        :param num_words:
        :return:
        """

        data_train = self.load_dataframe(train_file)
        data_test = self.load_dataframe(test_file)

        data_train, data_test = self.subset(data_train, data_test, section)

        if shuffle:
            data_train = self.shuffle_dataframe(data_train)
            data_test = self.shuffle_dataframe(data_test)

        print("TRAIN LEN: ", len(data_train))
        print("TEST LEN: ", len(data_test))

        # Separates training data
        x_train = self.retrieve_features(data_train)
        y_years, y_months, y_partys = self.retrieve_labels(data_train)

        # Parties
        y_train = y_partys

        # Separates testing data
        x_test = self.retrieve_features(data_test)
        y_years, y_months, y_partys = self.retrieve_labels(data_test)

        # parties
        y_test = y_partys

        # Prepare label data structure
        if self.load_tokenizers == False:
            self.x_tokenizer = Tokenizer()
            self.y_tokenizer = Tokenizer()

            self.x_tokenizer.fit_on_texts(x_train + x_test)
            self.y_tokenizer.fit_on_texts(y_train + y_test)

        # Prepare feature data structure
        if x_mode == "vectorize":
            x_train, x_test = self.build_vectors(x_train, x_test)
        elif x_mode == "index":
            x_train, x_test = self.build_indices(x_train, x_test, maxlen=feat_vec_len)

        if y_mode == "vectorize":
            y_train, y_test = self.build_vectors(y_train, y_test, is_label=True)
        elif y_mode == "index":
            y_train, y_test = self.build_indices(y_train, y_test, maxlen=1, pad=False, is_label=True)
            y_train = [idx[0] for idx in y_train]
            y_test = [idx[0] for idx in y_test]

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(
            y_test), self.x_tokenizer, self.y_tokenizer
