from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from datetime import datetime
from copy import deepcopy
import numpy as np
import argparse
import pickle
import json
from pathlib import Path

import os
import re
import csv
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="CLI for parsing arguments")

    implemented_methods = ['log_odds', 'tfidf', 'pmi', 'tfidf_pmi']

    corpus_default = "./data/default_corpus.json"

    help_corpus = f"A path to a json corpus in this format {corpus_default}."
    parser.add_argument(
        "--corpus",
        type=str,
        default=corpus_default,
        help=help_corpus,
    )

    comparison_corpus_default = "./data/default_comparison_corpus.json"

    help_corpus = f"A path to a json comparison_corpus in this format {corpus_default}. " \
                  f"You need this for the log_odd ratio. "

    parser.add_argument(
        "--comparison_corpus",
        type=str,
        default=comparison_corpus_default,
        help=help_corpus,
    )

    help_corpus = f"If you do not have a config.yml in the working directory or want to set your setting with the cli tool set this var to False. "

    parser.add_argument(
        "--config",
        type=bool,
        default=True,
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
        help="Frequency threshold for less frequent words (default: 1.0)",
    )

    parser.add_argument(
        "--method",
        type=str,
        default='tfidf_pmi',
        help=f"Choose a method from the list of implemented methods {implemented_methods}"
    )

    parser.add_argument(
        "--stop_words",
        type=str,
        default='english',
        help=f"Exclude stop_words from this list ['english']."
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
    max_count = np.array(np.max(X, axis=0)).reshape(-1,)
    upper = np.percentile(max_count, less_freq_than)
    min_count = (min_count >= lower)
    max_count = (max_count <= upper)
    return np.logical_and(min_count, max_count)


def calc_matrix(X, X_count):
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
        important_words = {'words': list(top_features)}
        if return_values:
            values = X[doc_idx, relevant_features]
            print(values.shape)
            values = values[top_indices]
            important_words['values'] = list(values)
        print(doc_idx, important_words['words'][:10], important_words['values'][:10])
        keyword_dict[labels[doc_idx]] = deepcopy(important_words)
    return keyword_dict


def create_keyword_dictionary(corpus: list,
                              labels: list,
                              min_df: int = None,
                              more_freq_than: int = 0,
                              less_freq_than: int = 100,
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

        background = kwargs['vectorizer'].transform(kwargs['comparison_corpus_d'])

        alpha = np.sum(background, axis=0) + np.sum(X, axis=0)

        X = log_odd_ratio(X, alpha)

        return X

    def pmi(X, **kwargs):
        pmi_matrix = calc_matrix(X, X)

        return pmi_matrix

    def tfidf_pmi(X, **kwargs):
        X_tfidf = tfidf(X, **kwargs)
        pmi_matrix = calc_matrix(X, X_tfidf)

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

def write_to_csv(output_file, data):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(["Word"] + list(data.keys()))
        all_words = set.union(*[set(w["words"]) for w in data.values()])

        # Iterate over the data and write rows
        for word in all_words:
            values = []
            for document, content in data.items():
                word_dict = dict(zip(content["words"], content["values"]))
                values.append(word_dict.get(word, None))
            writer.writerow([word] + values)
    return True

def main():
    args = parse_args()

    implemented_methods = ['tfidf', 'log_odds', 'pmi', 'tfidf_pmi']
    filename = generate_timestamp()

    def load_and_sync_config(config_path="config.yml"):
        config_path = Path(config_path)
        # 1) load YAML → dict (if empty, get {})
        cfg: dict = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Expected top‐level mapping in {config_path}, got {type(cfg)}")

        # 2) inject as argparse defaults
        parser = parse_args()
        parser.set_defaults(**cfg)

        # 3) parse real CLI → Namespace
        args = parser.parse_args()

        # 4) merge back into our dict
        #    Only overwrite keys for which the user actually passed something (i.e. non‐None or flags)
        for key, val in vars(args).items():
            # for flags (store_true), val==False means user didn’t pass it, so skip
            # for optional ints (min_df), val is either None or int; we overwrite even None
            is_flag   = key in ("only_words", "return_values")
            passed    = (is_flag and val) or (not is_flag and val is not None)
            if passed:
                cfg[key] = val

        # 5) write YAML back out
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return args

    output_dict = load_and_sync_config("./config.yml")

    corpus_path   = args.corpus
    language      = args.language
    min_df        = args.min_df
    more_freq_than= args.more_freq_than
    less_freq_than= args.less_freq_than
    method        = args.method
    return_values = args.return_values

    # Load the corpus and labels
    if isinstance(corpus_path, str):
        with open(corpus_path) as f:
            corpus_data = json.load(f)
            corpus = [doc['text'] for doc in corpus_data]
            labels = [doc['label'] for doc in corpus_data]


    assert method in implemented_methods, f"Please make sure that the method={method} is part of the implemented_methods={implemented_methods}."

    if isinstance(corpus, str):
        with open(corpus) as f:
            corpus = json.load(f)
    if isinstance(args.comparison_corpus, str):
        try:
            with open(args.comparison_corpus, "rb") as f:
                args.comparison_corpus_d = pickle.load(f)
            print("Corpus path:", corpus_path)
        except:
            with open(args.comparison_corpus) as f:
                args.comparison_corpus_d = json.load(f)


    # Do something with the parsed arguments
    print("Corpus path:", output_dict.corpus)
    print("Language:", language)
    print("Minimum document frequency:", min_df)
    print("More frequent than:", more_freq_than)
    print("Less frequent than:", less_freq_than)
    print("Method", method)

    argd = vars(args)        # now argd is a plain dict

    # pop off the ones you’ve already bound to locals...
    corpus        = argd.pop("corpus")
    method        = argd.pop("method")
    
    # then later:
    keyword_dict = create_keyword_dictionary(
        corpus=corpus,
        labels=labels,
        min_df=min_df,
        more_freq_than=more_freq_than,
        less_freq_than=less_freq_than,
        language=language,
        method=method,
        return_values=return_values,
        **argd            # <-- this is now a dict
    )
    try:
        del args.comparison_corpus_d
    except:
        pass

    output_directory = "./output/"

    os.makedirs(output_directory) if not os.path.exists(output_directory) else None

    output_file = f"{output_directory}{filename}_{method}.csv"

    write_to_csv(output_file, keyword_dict)

    output_directory = "./output_config/"

    os.makedirs(output_directory) if not os.path.exists(output_directory) else None

    with open(f"{output_directory}{filename}_{method}.json", "w") as f:
        json.dump(output_dict, f)



if __name__ == "__main__":
    main()
