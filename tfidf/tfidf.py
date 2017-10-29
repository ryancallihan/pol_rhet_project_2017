import numpy as np
import data_processing.data_processing as dp
from collections import Counter


class Tfidf:

    def __init__(self):

        self.token_appearance = Counter()
        self.num_docs = 0

    def build_model(self, texts):

        for text in texts:
            unique_tokens = set(text)
            for word in unique_tokens:
                self.token_appearance[word] = self.token_appearance[word] + 1
            self.num_docs = self.num_docs + 1
        print(self.token_appearance)
        print("TF*IDF model successfully built!")

    def counts(self, text):
        c = Counter(text)
        return c, len(text)

    def tf(self, token_count, doc_length):
        return token_count / doc_length

    def idf(self, num_docs, docs_containing_term):
        return np.log(1 + (num_docs - docs_containing_term) / docs_containing_term)

    def tf_idf(self, token_count, doc_length, num_docs, docs_containing_term):
        return self.tf(token_count, doc_length) * self.idf(num_docs, docs_containing_term)


def tfidf_analysis(sections, i):
    """
    Writes 50 highest scoring words from each political party per quarter
    :param sections:
    :param i:
    :return:
    """
    with open("tfidf_results/tfidf_scores.tsv", "w+") as results:
        results.write("time\t" +
                      "d_word\t" +
                      "d_score\t" +
                      'r_word\t' +
                      'r_score\t')
        results.close()


    print("WORKING ON SECTION:", i, save_path)

    data = dp.Data_Processing()
    x, y, x_test, y_test, x_tokenizer, y_tokenizer = data.run(
        train_file="../data/train_data/train_data.p",
        test_file="../data/test_data/test_data.p",
        section=i,
        shuffle=False)

    x = np.append(x, x_test)
    y = np.append(y, y_test)

    x = data.tweet_tokenizer(x)

    r_x = []
    d_x = []
    x_full = []

    for idx, tweet in enumerate(x):
        if y[idx] == "R":
            r_x.append(tweet)
        else:
            d_x.append(tweet)
        x_full.append(tweet)

    t = Tfidf()
    t.build_model(x_full)

    r_tfidf = dict()
    d_tfidf = dict()

    for tweet in r_x:
        counts, length = t.counts(tweet)
        for word in tweet:
            score = t.tf_idf(counts[word], length, t.num_docs, t.token_appearance[word])
            r_tfidf[word] = r_tfidf.get(word, 0) + score

    for tweet in d_x:
        counts, length = t.counts(tweet)
        for word in tweet:
            score = t.tf_idf(counts[word], length, t.num_docs, t.token_appearance[word])
            d_tfidf[word] = d_tfidf.get(word, 0) + score

    r_words = [item[0] for item in r_tfidf.items()]
    r_scores = [item[1] for item in r_tfidf.items()]
    r_top_words = [(r_words[idx], r_scores[idx]) for idx in list(np.argsort(r_scores))][::-1]

    d_words = [item[0] for item in d_tfidf.items()]
    d_scores = [item[1] for item in d_tfidf.items()]
    d_top_words = [(d_words[idx], d_scores[idx]) for idx in list(np.argsort(d_scores))][::-1]

    with open("tfidf_results/tfidf_scores.tsv", "a") as results:
        for idx in range(50):
            results.write("\n%s\t%s\t%.5f\t%s\t%.5f" %
                          (save_path, d_top_words[idx][0], d_top_words[idx][1], r_top_words[idx][0],
                           d_top_words[idx][1]))
        results.close()

    print(save_path, "FINISHED")


if __name__ == "__main__":

    sections = ["2015_1q", "2015_2q", "2015_3q", "2015_4q", "2016_1q", "2016_2q", "2016_3q", "2016_4q", "2017_1q",
                "2017_2q"]

    for idx, save_path in enumerate(sections):

        i = idx + 1

        tfidf_analysis(save_path, i)

