import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score
from tf_models.tf_rnn_embeddings.config import DefaultConfig
from tensorflow.python.framework import ops
import data_processing.data_processing as dp
from tf_models.tf_rnn_embeddings.model import Model, Phase
from keras.preprocessing.text import Tokenizer


def generate_instances(
        data,
        labels_data,
        n_word,
        n_label,
        max_timesteps,
        batch_size=128):
    n_batches = len(data) // batch_size

    # We are discarding the last batch for now, for simplicity.
    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            n_label),
        dtype=np.float32)
    lengths = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)
    words = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)

    for batch in range(n_batches):
        for idx in range(batch_size):
            word = data[(batch * batch_size) + idx]
            # Add label distribution

            label = labels_data[(batch * batch_size) + idx]
            index = np.nonzero(label)[0][0]
            labels[batch, idx, index] = 1

            # Sequence
            timesteps = min(max_timesteps, len(word))

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            # Word characters
            words[batch, idx, :timesteps] = word[:timesteps]

    return (words, lengths, labels, n_word)


def train_model(config, train_batches, validation_batches, word_vector, save_path, from_saved=False):
    """
    Trains RNN model
    :param config:
    :param train_batches:
    :param validation_batches:
    :param word_vector:
    :param save_path:
    :param from_saved:
    :return:
    """
    train_batches, train_lens, train_labels, n_word = train_batches
    validation_batches, validation_lens, validation_labels, n_word = validation_batches

    if not from_saved and save_path == "2015_1q":
        with open("tf_training_hist/training_history.tsv", "w") as history:
            history.write("epoch\t" +
                          "train_loss\t" +
                          "val_loss\t" +
                          "val_acc\t" +
                          "f1score\t" +
                          "recall\t" +
                          "precision\t" +
                          "time\t")
            history.close()

    ops.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:
        with tf.variable_scope(save_path, reuse=False):
            train_model = Model(
                config=config,
                batch=train_batches,
                lens_batch=train_lens,
                label_batch=train_labels,
                word_vector=word_vector,
                phase=Phase.Train)

        with tf.variable_scope(save_path, reuse=True):
            validation_model = Model(
                config=config,
                batch=validation_batches,
                lens_batch=validation_lens,
                label_batch=validation_labels,
                word_vector=word_vector,
                phase=Phase.Validation)

        # Initialize Saver
        saver = tf.train.Saver()

        if from_saved:
            print("LOADING FROM SAVED MODEL>>>>")
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, os.path.join("tf_saved_models", save_path, save_path))
            print("Model restored.")
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            train_loss = 0.0
            validation_loss = 0.0
            accuracy = 0.0
            recall = 0.0
            precision = 0.0
            f_score = 0.0
            # Train on all batches.
            for batch in range(train_batches.shape[0]):
                loss, _ = sess.run([train_model.loss, train_model.train_op], {
                    train_model.x: train_batches[batch], train_model.y: train_labels[batch]})
                train_loss += loss
            # validation on all batches.
            for batch in range(validation_batches.shape[0]):
                loss, acc, y_true, y_pred = sess.run(
                    [validation_model.loss, validation_model.accuracy, validation_model.hp_labels,
                     validation_model.labels], {
                        validation_model.x: validation_batches[batch], validation_model.y: validation_labels[batch]})

                validation_loss += loss
                accuracy += acc
                recall += recall_score(y_true, y_pred, average='micro')
                precision += precision_score(y_true, y_pred, average='micro')
                f_score += f1_score(y_true, y_pred, average='micro')

            train_loss /= train_batches.shape[0]
            validation_loss /= validation_batches.shape[0]
            accuracy /= validation_batches.shape[0]
            recall /= validation_batches.shape[0]
            precision /= validation_batches.shape[0]
            f_score /= validation_batches.shape[0]

            print(
                "epoch %d - train loss: %.3f, validation loss: %.3f, validation acc: %.3f, f1 score: %.3f, recall: "
                "%.3f, precision: %.3f, time: %s" %
                (epoch + 1, train_loss, validation_loss, accuracy * 100, f_score * 100, recall, precision, save_path))
            with open("tf_training_hist/training_history.tsv", "a") as history:
                history.write("\n%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s" %
                              (epoch + 1, train_loss, validation_loss, accuracy * 100,
                               f_score * 100, recall, precision, save_path))
            history.close()

            # Saves model
            save_to_disk = saver.save(sess=sess, save_path=os.path.join("tf_saved_models", save_path, save_path))
            print("MODEL SAVED>>>>")
        sess.close()


def predict(config, validation_batches, word_vector, save_path, texts=None, data_subset=None, pred_type="tweet"):
    """
    Predicts scores of inputs
    :param config:
    :param validation_batches:
    :param word_vector:
    :param save_path:
    :param texts:
    :param data_subset:
    :param pred_type:
    :return:
    """
    if data_subset is None:
        data_subset = save_path

    validation_batches, validation_lens, validation_labels, n_word = validation_batches


    # Binary
    with open("".join(["tf_prediction_results/tweet_prediction_",save_path,"_",pred_type,".tsv"]), "w", encoding="utf-8") \
            as history:

        history.write("text\t" +
                      "hp_label\t" +
                      "pred_label\t" +
                      "D_prob\t" +
                      "R_prob\t" +
                      "time\t")



        history.close()

    # Resets graph
    ops.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:

        with tf.variable_scope(save_path, reuse=None):
            validation_model = Model(
                config=config,
                batch=validation_batches,
                lens_batch=validation_lens,
                label_batch=validation_labels,
                word_vector=word_vector,
                phase=Phase.Validation)

        # Initialize Saver
        saver = tf.train.Saver()

        print("LOADING FROM SAVED MODEL>>>>")
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, os.path.join("tf_saved_models", save_path, save_path))
        print(save_path, "Model restored.")

        # Train on all batches.

        tweet_idx = 0
        with open("".join(["tf_prediction_results/tweet_prediction_", save_path, "_", pred_type, ".tsv"]), "a",
                  encoding="utf-8") as history:
            accuracy = 0.0
            for batch in range(validation_batches.shape[0]):
                loss, acc, y_true, y_pred, logits = sess.run(
                    [validation_model.loss, validation_model.accuracy, validation_model.hp_labels,
                     validation_model.labels, validation_model.logits], {
                        validation_model.x: validation_batches[batch], validation_model.y: validation_labels[batch]})
                accuracy += acc


                for t_idx in range(config.batch_size):


                    # Binary
                    history.write("\n%s\t%d\t%d\t%.10f\t%.10f\t%s" %
                                  (texts[tweet_idx], y_true[t_idx], y_pred[t_idx], logits[t_idx][0],
                                   logits[t_idx][1], save_path))

                    tweet_idx += 1

        history.close()

    sess.close()
    del sess, validation_model


def tweet_predict(save_path, i):
    """
    Creates TSV with scores of all tweets
    :param save_path:
    :param i:
    :return:
    """
    print("TRAINING SECTION:", i, save_path)
    config = DefaultConfig()

    data = dp.Data_Processing(load_tokenizers=True)
    x_train, y_train, x_test, y_test, x_tokenizer, y_tokenizer = data.run(
        train_file="../../data/train_data/train_data.p",
        test_file="../../data/test_data/test_data.p",
        section=i,
        x_mode="index",
        y_mode="vectorize",
        shuffle=False)

    tweets_train, _, tweets_test, _, _, _ = data.run(
        train_file="../../data/train_data/train_data.p",
        test_file="../../data/test_data/test_data.p",
        section=i,
        shuffle=False)

    num_words = len(x_tokenizer.word_index) + 1
    num_classes = len(y_tokenizer.word_index.items())
    print("NUM WORDS: ", num_words, " NUM CLASSES: ", num_classes)

    w2v_layer = data.create_embedding_layer('../../trained_w2v/twitter2vec.w2v')

    # TODO STILL ONLY VALIDATION BATCHES?
    validation_batches = generate_instances(
        x_test,
        y_test,
        num_words,
        num_classes,
        config.max_timesteps,
        batch_size=config.batch_size)

    # Train the model
    predict(config, validation_batches, w2v_layer, save_path, tweets_test)
    print("FINISHED:", save_path, "TWEET PREDICTION\n\n")


def word_predict(save_path, i):
    """
    Creates TSV with scores of all vocabulary.
    :param save_path:
    :param i:
    :return:
    """
    print("TRAINING SECTION:", (i), save_path)
    config = DefaultConfig()

    data = dp.Data_Processing(load_tokenizers=True)
    x_train, y_train, x_test, y_test, x_tokenizer, y_tokenizer = data.run(
        train_file="../../data/train_data/train_data.p",
        test_file="../../data/test_data/test_data.p",
        section=i,
        y_mode="vectorize",
        shuffle=False)

    get_counts = Tokenizer()
    get_counts.fit_on_texts(x_test)

    words = [word[0] for word in get_counts.word_index.items()]

    word_vectors = data.pad_sequences(data.vectorize_text(words), max_len=30)

    labels = [[0, 1] for label in range(len(words))]

    num_words = len(x_tokenizer.word_index) + 1
    num_classes = len(y_tokenizer.word_index.items())
    print("LEN X: ", len(word_vectors))

    w2v_layer = data.create_embedding_layer('../../trained_w2v/twitter2vec.w2v')

    validation_batches = generate_instances(
        word_vectors,
        labels,
        num_words,
        num_classes,
        config.max_timesteps,
        batch_size=config.batch_size)

    # Train the model
    predict(config, validation_batches, w2v_layer, save_path, words, pred_type="words")
    print("FINISHED:", save_path, "WORD PREDICTION\n\n")


def train(save_path, i, from_saved=False):
    """
    Trains RNN model with given save_path. Can resume training from saved model.
    :param save_path:
    :param i:
    :return:
    """
    print("TRAINING SECTION:", i, save_path)
    config = DefaultConfig()

    data = dp.Data_Processing(load_tokenizers=True)
    x_train, y_train, x_test, y_test, x_tokenizer, y_tokenizer = data.run(
        train_file="../../data/train_data/train_data.p",
        test_file="../../data/test_data/test_data.p",
        section=i,
        x_mode="index",
        y_mode="vectorize")

    num_words = len(x_tokenizer.word_index) + 1
    num_classes = len(y_tokenizer.word_index.items())

    # TODO CHECK INDEX LIST

    w2v_layer = data.create_embedding_layer('../../trained_w2v/twitter2vec.w2v')

    # Generate batches
    train_batches = generate_instances(
        x_train,
        y_train,
        num_words,
        num_classes,
        config.max_timesteps,
        batch_size=config.batch_size)
    validation_batches = generate_instances(
        x_test,
        y_test,
        num_words,
        num_classes,
        config.max_timesteps,
        batch_size=config.batch_size)

    # Train the model
    train_model(config, train_batches, validation_batches, w2v_layer, save_path, from_saved=from_saved)
    print("FINISHED:", save_path, "TRAINING\n\n")


if __name__ == "__main__":

    sections = ["2015_1q", "2015_2q", "2015_3q", "2015_4q", "2016_1q", "2016_2q", "2016_3q", "2016_4q", "2017_1q",
                "2017_2q"]


    for idx_model, model in enumerate(sections):
        i_data = idx_model+1

        # tweet_eval(model, i_data)
        # word_eval(model, i_data)
        train(model, i_data, from_saved=False)
