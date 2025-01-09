# Comparing Keyword Importance Across Texts

## Description
This method identifies and ranks the most important words in a collection of documents, such as articles, speeches, or social media posts, by analyzing their frequency and uniqueness within each document. 
Using measures like TF-IDF, PMI, and Log Odds Ratio, it highlights terms that are especially relevant to a specific document while contrasting them with others in the collection. 
This approach is ideal for uncovering key themes, comparing language use across texts, and tracking shifts in terminology or public discourse over time, making it a valuable tool for summarizing content or analyzing trends.


| | TF-IDF | Log Odds Ratio| PMI |
|:---|:---:|:---:|:---:|
| Definition | Measures the importance of a term in a document not only by frequent usage but also through the absence of use in other documents. | Quantifies the increase of the relative importance of a term for a document in comparison to all other documents. | Measures the association between a term and a document, indicating a dependency. |
| When to use?       | Finding terms that are characteristic for a document and are only used by a subset of other documents. | Finding terms that have higher relevance for a certain document. | Finding terms that are characteristic for a document and are seldom used by other documents. |
| Interpretability | High Scores: indicate greater importance of the term within the document. | Positive: indicates association with the document. Negative: indicates low importance of term for the document. | High Scores: Indicate strength of association between term and document. Low Scores: indicate disassociation between term and the document. |

## Keywords
Comparative Analysis, Keyword Extraction, Word Importance, Log Odd Ratio,

## Use Case
A social scientist studying climate change discourse on Twitter over time. By extracting and comparing keywords, it reveals emerging terms (e.g., "carbon neutrality"), diminishing terms (e.g., "global warming"), and stable terms (e.g., "climate crisis"), offering insights into evolving public conversations and priorities.

## Directory Structure
The method consists of one file [keyword_extraction.py](keyword_extraction.py).
The data used for the demo run is saved in the [/data/](data/) folder.
Once the method is finished the method will create the following folder structure and output.
In the /ouptut/ folder you find the csv with the word importance scores.
In /output_config/ you find a json specify all the parameters used to produce the csv.

```
.
├── keyword_extraction.py
├── config.json
├── output
│   └── the-time-you-ran-the-code_pmi.csv
├── output_config
│   └── the-time-you-ran-the-code_pmi.json
├── data
│   └── default_corpus.json
│   └── default_comparison_corpus.json
├── README.md
└── requirements.txt

```


# Setup
## Environment Setup
Install Python v>=3.9 (preferably through Anaconda).

Download the repository with or directly copy the raw code from [keyword_extraction.py](keyword_extraction.py), and requirements.txt
```
git clone https://git.gesis.org/bda/keyword_extraction.git
```



## Installing Dependencies
- Installing all the packages and libraries with specific versions required to run this method

  ```
  pip install -r requirements.txt
  ```

- The user should be able to reuse the method following the information provided


# Usage
## Input Data (Digital Behavior Data datasets)
The method handles digital behavior data, including social media posts, comments, search queries, clickstream text (e.g., website titles), forum threads, and open-text survey responses.

## Sample Input Data
The corpus data used in the script is stored in JSON format at [data/default_corpus.json](data/default_corpus.json) and looks something like this: 

```
{
    'Document A': "This is the liberal solution: All text is good aswell as bad. The good one has to take his own position . We are the liberal ones . Not the center nor the progressive ones.",
    'Document B': "This is the center solution: They are bad not good, if everyone remains on his own position we are all alone which is bad . We are the center ones . Not the progressive nor the liberal ones .",
    'Document C': "This is the progressive solution: Another groups position is the problem. They dont move from their position . We are the progressive ones . Not the liberal nor the center ones . "
}
```

## Sample Output
This file can be run from the Terminal with:


```
python keyword_extraction.py -help
```

This call will return the list of parameters you can specify.
There you can find an explanation for all and their respective functionality.
Once you are familiar with the possible parameters you can run the code like this:

```
python keyword_extraction.py -method pmi -corpus /path/to/your_corpus.json
```

The method will produce a csv in the following form:

|Words | Document A | Document B | Document C|
|:---  | :---:   | :---:   | ---:|
|example | pmi_a | pmi_b | pmi_c |


Moreover, in the /config_output/ you find a json file that saved all the used parameters for the resulting table.

```
{
    "corpus": "/path/to/your_corpus.json",
    "comparison_corpus": "",
    "language": "english",
    "min_df": null,
    "more_freq_than": 0,
    "less_freq_than": 100,
    "method": "pmi",
    "only_words": true,
    "return_values": true
}

```

## How to Use

```
python keyword_extraction.py -method pmi -corpus /path/to/your_corpus.json -more_freq_than 80
```
With this call you calculate the Pointwise Mutual Information for words that appear more frequent than 80% of all the words found in the documents. So, just words that appear very often.


```
python keyword_extraction.py -method tfidf -corpus /path/to/your_corpus.json -min_df 2
```
Calling the method like this you will create the importance scores on the basis of TFIDF (Term frequency Inverse Document Frequency). This method weighs words higher when they dont appear in all documents. Meaning a word that only appears in this document is probably more important for this document than a word that is shared across all documents. With 'min_df' we specify that the words should at least appear in 2 documents. Thus, we exclude all words that only appear in one document.

```
python keyword_extraction.py -method pmi_tfidf -corpus /path/to/your_corpus.json -less_freq_than 20
```
If you run this code you calculate the Pointwise Mutual Information for the tfidf scores of the words. This can be beneficial as it already accounts for the importance of a word for a specific document when weighting it. Furthermore, we exclude the 20% of words that appear less often.

```
python keyword_extraction.py -method log_odds -corpus /path/to/your_corpus.json -comparison_corpus /path/to/your_comparison_corpus.json 
```
To calculate the Log Odd Ratio we need to specify a comparison corpus. This corpus should be unbiased as it is used to quantify how often certain words appear under normal circumstances. With this information, we can alleviate the influence of noise when we calculate the importance of our words for the corpus.

Or use the config flag to load the configuration from the [config.json](config.json) file

```
python keyword_extraction.py -config True
```

# Specifics
## Contact Details
Stephan.Linzbach@gesis.org

