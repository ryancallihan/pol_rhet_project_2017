import logging
import multiprocessing
import os
import re
import gensim
import nltk
import gensim.models.word2vec as w2v
import numpy as np
import pandas as pd


def prepare_text(text):
    """
    Prepares text to be read by w2v
    :param text:
    :return:
    """
    print("preparing text...")
    tokenizer = nltk.TweetTokenizer(preserve_case=False)
    tokenized_sentences = []
    for raw_sentence in text:
        if not isinstance(raw_sentence, str):
            print("NOT A STRING")
            raw_sentence = str(raw_sentence)
        if len(raw_sentence) > 0:
            cleaned_sentence = re.sub("[^a-zA-Z#@]", " ", raw_sentence)
            tokenized_sentences.append(tokenizer.tokenize(cleaned_sentence))
    print("Example Tokenized Sent: ", tokenized_sentences[10])
    print(sum([len(sentence) for sentence in tokenized_sentences]), " tokens in the corpus")
    return tokenized_sentences, tokenizer


def train_word2vec(processed_text, num_features=300, min_word_count=3, context_size=7):
    """
    Trained w2v on preprocessed text
    :param processed_text:
    :param num_features:
    :param min_word_count:
    :param context_size:
    :return:
    """
    num_workers = multiprocessing.cpu_count()
    downsampling = 1e-3
    seed = 1

    twitter2vec = w2v.Word2Vec(
        sg=1,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling,
        iter=50
    )

    twitter2vec.build_vocab(processed_text)

    print("Word2Vec vocabulary length:", twitter2vec.corpus_count)

    twitter2vec.train(
        processed_text,
        total_examples=twitter2vec.corpus_count,
        epochs=twitter2vec.iter
    )

    if not os.path.exists("trained_w2v_visual"):
        os.makedirs("trained_w2v_visual")
    twitter2vec.save(os.path.join("trained_w2v_visual", "twitter2vec_visual.w2v"))


if __name__ == "__main__":

    tweets = pd.read_csv("../data_processing/cong_data.csv")

    tweets_processed = prepare_text(tweets["tweet"])

    w2v_path = os.path.join("..", "word2vec", "trained_w2v_visual", "twitter2vec_visual.w2v")
    words = dict()

    for t in tweets_processed[0]:
        for word in t:
            words[word] = 1

    words = [word for word in words.keys()]
    print("LEN OF WORDS UNIQ", len(words))

    w2v_model = gensim.models.Word2Vec.load(w2v_path)
    num_words = len(words)
    print("NUM WORDS:", num_words)
    w2v_size = w2v_model.vector_size
    print("WORD2VEC SHAPE:", w2v_size)
    w2v_embeddings = np.zeros((num_words, w2v_size))

    for index, word in enumerate(words):
        try:
            print(index)
            w2v_embeddings[index] = w2v_model[word]
        except KeyError:
            continue

    w2v_list_new = []
    words_new = []
    for idx, vector in enumerate(w2v_embeddings):
        if vector[0] != 0.0 and vector[3] != 0.0:
            w2v_list_new.append(vector)
            words_new.append(words[idx])

    print("Len of words", len(words))
    print("Len of words new", len(words_new))

    with open('w2v_vectors.tsv', 'w', encoding='utf-8') as data:
        for word in w2v_list_new:
            for point in word:
                data.write(str(point))
                data.write("\t")
            data.write("\n")
        print("FINISHED WRITING VECTORS")
        data.close()

    with open('w2v_words.tsv', 'w', encoding='utf-8') as data:
        data.write("WORDS")
        for word in words_new:
            data.write("\n" + word)
        print("FINISHED WRITING WORDS")
        data.close()
