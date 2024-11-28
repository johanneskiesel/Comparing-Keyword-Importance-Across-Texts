# Comparing Keyword Importance Across Texts

## Description
Given a corpus containing different documents.
These documents can be news articles, forum entries, websites, or a manifesto.
The method can calculate the document-specific importance of a word in four different ways (tfidf, pmi, pmi+tfidf and log odds ratio).
For example, you can use the 50 most important terms for a specific document to describe its topic.
The resulting information is useful if you want to analyze the language of political and social groups or summarize a document.


| | TF-IDF | Log Odds Ratio| PMI |
|:---|:---:|:---:|:---:|
| Definition | Measures the importance of a term in a document not only by frequent usage but also through the absence of use in other documents. | Quantifies the increase of the relative importance of a term for a document in comparison to all other documents. | Measures the association between a term and a document, indicating a dependency. |
| When to use?       | Finding terms that are characteristic for a document and are only used by a subset of other documents. | Finding terms that have higher relevance for a certain document. | Finding terms that are characteristic for a document and are seldom used by other documents. |
| Interpretability | High Scores: indicate greater importance of the term within the document. | Positive: indicates association with the document. Negative: indicates low importance of term for the document. | High Scores: Indicate strength of association between term and document. Low Scores: indicate disassociation between term and the document. |

## Keywords
Comparative Analysis, Keyword Extraction, Word Importance, Log Odd Ratio,

## Typical use cases in the social sciences

### Use Case: Identifying Trends in Social Media Discussions
#### Scenario
Imagine you are a social scientist studying public discourse on climate change over time. You want to identify emerging trends and changes in how people discuss the topic on Twitter.
#### Method Application

Using this method, you analyze tweets from different periods (e.g., monthly or yearly datasets). For each period you perform: \
    1. **Keyword Extraction**: Apply metrics like TF-IDF or Log Odds Ratio to identify terms uniquely important in that period's tweets. \
    2. **Trend Detection**: Compare the importance scores of keywords across periods to see which terms are gaining or losing prominence. 

#### Outcome
The analysis reveals: \
    -**Emerging Keywords**: For example, new terms like carbon neutrality or greenwashing might appear in recent datasets, indicating shifts in public focus. \
    -**Diminishing Keywords**: Older terms like global warming might show reduced importance, reflecting changes in terminology or framing. \
    -**Stable Keywords**: Terms like climate crisis might maintain steady importance, showing consistency in discourse.

#### Why is it Useful?
This trend analysis helps identify how public conversations evolve, providing insights into societal priorities, policy impacts, or advocacy success. It can also guide future research or communication strategies on climate issues.

## Directory Structure
The method consists of one file keyword_extraction.py.
Once the method is finished the method will create the following folder structure and output.
In the /ouptut/ folder you find the csv with the word importance scores.
In /output_config/ you find a json specify all the parameters used to produce the csv.

```
.
├── keyword_extraction.py
├── output
│   └── the-time-you-ran-the-code_pmi.csv
├── output_config
│   └── the-time-you-ran-the-code_pmi.json
├── output.csv
├── README.md
└── requirements.txt

```


# Setup
## Environment Setup
Download the repository with or directly copy the raw code from keyword_extraction.py, and requirements.txt
```
git clone https://git.gesis.org/bda/keyword_extraction.git
```

Then navigate in your project folder and run 
```
pip install -r requirements.txt
```

## Installing Dependencies
- Installing all the packages and libraries with specific versions required to run this method
- The user should be able to reuse the method following the information provided


# Usage
## Input Data (Digital Behavior Data datasets)
- [German Federal Elections](https://search.gesis.org/research_data/ZA7721?doi=10.4232/1.13789)
- [TweetsCov19](https://data.gesis.org/tweetscov19/#dataset)
- [Call me sexist but](https://data.gesis.org/sharing/#!Detail/10.7802/2251)
- [Incels Forum Data](https://search.gesis.org/research_data/SDN-10.7802-2485?doi=10.7802/2485)

## Sample Input Data
The corpus data should look something like this: 

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


# Specifics
## Contact Details
Stephan.Linzbach@gesis.org

