from datetime import datetime
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import argparse
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser(description="CLI for parsing arguments")

    implemented_methods = ['log_odds', 'tfidf', 'pmi', 'tfidf_pmi']

    corpus_default = {'Class A': "This is it the liberal solution: "
                                 "All the text is good aswell as bad. The good one has to take its own position ."
                                 "We are the liberal ones ."
                                 "Not the center nor the progressive ones.",
                      'Class B': "This is it the center solution: They are bad not good if everyone is on "
                                 "its own position we are all alone which is bad ."
                                 "We are the center ones ."
                                 "Not the progressive nor the liberal ones .",
                      'Class C': "This is it the progressive solution: "
                                 "Another groups position is the problem they dont move from their position ."
                                 "We are the progressive ones ."
                                 "Not the liberal nor the center ones . "}

    help_corpus = f"A path to a json corpus in this format {corpus_default}."
    parser.add_argument(
        "--corpus",
        type=str,
        default=corpus_default,
        help=help_corpus,
    )

    corpus_default = {'Class A': "All the text",
                      'Class B': "What do they write",
                      'Class C': "And another group"}

    help_corpus = f"A path to a json comparison_corpus in this format {corpus_default}. You need this for the log_odd ratio."
    parser.add_argument(
        "--comparison_corpus",
        type=str,
        default=corpus_default,
        help=help_corpus,
    )

    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help="Language (default: english)",
    )

    parser.add_argument(
        "--min_df",
        type=int,
        default=None,
        help="Minimum document frequency (default: 1)",
    )

    parser.add_argument(
        "--more_freq_than",
        type=int,
        default=0,
        help="Frequency threshold for more frequent words (default: 0)",
    )

    parser.add_argument(
        "--less_freq_than",
        type=int,
        default=100,
        help="Frequency threshold for less frequent words (default: 100)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default='log_odds',
        help=f"Choose a method from the list of implemented methods {implemented_methods}"
    )

    parser.add_argument(
        "--only_words",
        type=bool,
        default=True,
        help=f"Exclude numbers, urls, and everything that is not alphabetic."
    )

    parser.add_argument(
        "--return_values",
        type=bool,
        default=True,
        help=f"Use this parameter if you want the associated values of the respective method to be returned."
    )
    return parser.parse_args()


def get_count_vectorizer(min_df, stop_words, **kwargs):
    return CountVectorizer(strip_accents='unicode', min_df=min_df, stop_words=stop_words)


def get_tfidf_vectorizer(min_df, stop_words, **kwargs):
    return TfidfVectorizer(strip_accents='unicode', min_df=min_df, stop_words=stop_words)


def transform_to_tfidf(count_matrix):
    tfidf_transformer = TfidfTransformer()
    return tfidf_transformer.fit_transform(count_matrix)


def get_count_vectorizer_and_matrix(corpus, min_df, stop_words, **kwargs):
    return get_matrix(get_count_vectorizer(min_df=min_df,
                                           stop_words=stop_words,
                                           **kwargs),
                      corpus=corpus)


def get_matrix(vectorizer, corpus):
    return np.asarray(vectorizer.fit_transform(corpus).todense()), vectorizer, vectorizer.get_feature_names_out()


def transform(vectorizer, corpus):
    return vectorizer.transform(corpus)


def remove_non_words(features):
    is_word = np.array([not re.search('\d', f) for f in features])
    return is_word


def calc_upper_and_lower_freq(X, more_freq_than, less_freq_than):
    min_count = np.array(np.min(X, axis=0)).reshape(-1, )
    lower = np.percentile(min_count, more_freq_than)
    # min_count = np.array(np.min(X.todense(), axis=0)).reshape(-1,)
    upper = np.percentile(min_count, less_freq_than)
    min_count = (min_count >= lower)
    max_count = (min_count <= upper)
    return (np.logical_and(min_count, max_count))


def calc_pmi_matrix(X, X_count):
    total_docs = len(X_count)
    term_count = np.array(X_count.sum(axis=0)).flatten()
    doc_count = np.array(X_count.sum(axis=1)).flatten()

    doc_count[doc_count == 0] = 1

    pmi_matrix = np.log2((X * total_docs) / (np.outer(doc_count, term_count)))
    return pmi_matrix


def log_odd_ratio(X, alpha_w):
    output_x = []

    for i in range(X.shape[0]):
        y_i_w = X[i, :]
        y_j_w = np.sum(X, axis=0) - y_i_w

        n_i = np.sum(y_i_w)
        n_j = np.sum(y_j_w)
        alpha_0 = np.sum(alpha_w)

        i_importance = (y_i_w + alpha_w) / (n_i + alpha_0 - y_i_w - alpha_w)
        j_importance = (y_j_w + alpha_w) / (n_j + alpha_0 - y_j_w - alpha_w)

        explainability_i = 1 / (y_i_w + alpha_w)
        explainability_j = 1 / (y_j_w + alpha_w)

        log_odd_i = np.log(i_importance) - np.log(j_importance)
        variance_i_j = explainability_i + explainability_j

        log_odd_z_score_i = log_odd_i / np.sqrt(variance_i_j)
        output_x.append(np.array(log_odd_z_score_i))

    return np.vstack(output_x)


def features_to_exclude(X, features, more_freq_than, less_freq_than, param):
    only_words = param['only_words']

    freq_exclusion = calc_upper_and_lower_freq(X, more_freq_than=more_freq_than, less_freq_than=less_freq_than)
    print(freq_exclusion)
    if only_words:
        word_exclusion = remove_non_words(features)
        return np.logical_and(freq_exclusion, word_exclusion)

    return freq_exclusion


def fill_keyword_dict(X, labels, features, relevant_features, return_values, **kwargs):

    top_n_features_indices = np.argsort(-X[:, relevant_features], axis=1)
    keyword_dict = {}

    # Print the top N features for each document
    for doc_idx, top_indices in enumerate(top_n_features_indices):
        top_features = features[relevant_features][top_indices]
        important_words = {}
        important_words['words'] = list(top_features)
        if return_values:

            values = X[doc_idx, relevant_features]
            print(values.shape)
            values = values[top_indices]
            important_words['values'] = list(values)
        keyword_dict[labels[doc_idx]] = important_words.copy()
    return keyword_dict


def create_keyword_dictionary(corpus: list[str],
                              labels: list[str],
                              min_df: int = None,
                              more_freq_than: int = 95,
                              less_freq_than: int = 99,
                              language: str = 'english',
                              method: str = 'log_odds',
                              return_values: bool = True,
                              **kwargs):
    """
    labels : a ordered list of labels for each document in the corpus
    corpus : a list of documents
    """

    def tfidf(X, **kwargs):
        X_tfidf = np.asarray(transform_to_tfidf(X).todense())
        return X_tfidf

    def log_odds(X, **kwargs):
        assert 'comparison_corpus' in kwargs, "Include a comparison corpus to account for noise in the log_odds_ratio."
        assert 'vectorizer' in kwargs, "Include a vectorizer to transform the comparison with given vocabulary."

        background = transform(kwargs['vectorizer'], kwargs['comparison_corpus'])

        alpha = np.sum(background, axis=0) + np.sum(X, axis=0)

        X = log_odd_ratio(X, alpha)

        return X

    def pmi(X, **kwargs):
        pmi_matrix = calc_pmi_matrix(X, X)

        return pmi_matrix

    def tfidf_pmi(X, **kwargs):
        X_tfidf = tfidf(X, **kwargs)
        pmi_matrix = calc_pmi_matrix(X, X_tfidf)

        return pmi_matrix

    assert len(labels) == len(
        corpus), f"Please make sure that corpus (N={len(corpus)}) has the same size as labels (M={len(labels)})."
    assert len(set(labels)) == len(labels), f"Please make sure that labels is a non repeating list"

    if not min_df:
        print("Only looking at keywords that are available in all documents.")
        min_df = len(labels)

    X, vectorizer, features = get_count_vectorizer_and_matrix(corpus=corpus,
                                        min_df=min_df,
                                        stop_words=language)
    print(X.shape)
    print(vectorizer)

    kwargs['vectorizer'] = vectorizer

    relevant_features = features_to_exclude(X,
                                           features,
                                           more_freq_than=more_freq_than,
                                           less_freq_than=less_freq_than,
                                           param=kwargs)
    X = locals()[method](X=X,
                         **kwargs)

    print(features)
    keyword_dict = fill_keyword_dict(X=X,
                                     relevant_features=relevant_features,
                                     labels=labels,
                                     features=features,
                                     return_values=return_values,
                                     **kwargs)
    return keyword_dict


def generate_timestamp():
    # Get the current timestamp
    current_time = datetime.now()

    # Format the timestamp as a string suitable for a filename
    timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Construct the filename
    filename = f"{timestamp_str}"
    return filename

def main():
    args = parse_args()

    implemented_methods = ['tfidf', 'log_odds', 'pmi', 'tfidf_pmi']

    args = vars(args)
    output_dict = args.copy()
    corpus = args.pop('corpus')
    language = args.pop('language')
    min_df = args.pop('min_df')
    more_freq_than = args.pop('more_freq_than')
    less_freq_than = args.pop('less_freq_than')
    method = args.pop('method')
    return_values = args.pop('return_values')
    assert method in implemented_methods, f"Please make sure that the method={method} is part of the implemented_methods={implemented_methods}."
    # Do something with the parsed arguments
    print("Corpus path:", corpus)
    print("Language:", language)
    print("Minimum document frequency:", min_df)
    print("More frequent than:", more_freq_than)
    print("Less frequent than:", less_freq_than)
    print("Method", method)

    keyword_dict = create_keyword_dictionary(corpus=list(corpus.values()),
                                                   labels=list(corpus.keys()),
                                                   min_df=min_df,
                                                   more_freq_than=more_freq_than,
                                                   less_freq_than=less_freq_than,
                                                   language=language,
                                                   method=method,
                                                   return_values=return_values,
                                                   **args)

    filename = generate_timestamp()

    output_directory = "./output/"

    os.makedirs(output_directory) if not os.path.exists(output_directory) else None

    output_dict['keywords'] = keyword_dict
    with open(f"{output_directory}{filename}_{method}.json", "w") as f:
        json.dump(output_dict, f)

if __name__ == "__main__":
    main()

