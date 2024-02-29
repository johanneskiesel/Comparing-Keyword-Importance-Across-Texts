# Comparative Keyword Importance

## Description
Given a corpus containing text written by specific groups (right/left leaning; pro/contra climate change; feminist vs. manosphere), the method can perform the calculation of importance scores per word and group in four different ways (tfidf, pmi, pmi+tfidf and log odds ratio).
The resulting information is useful in text based analysis.


| | TF-IDF | Log Odds Ratio| PMI |
|:---|:---:|:---:|:---:|
| Definition | Measures the importance of a term in a group not only by frequent usage but also through the absence of use in other groups. | Quantifies the increase of the relative importance of a term for a group in comparison to all other groups. | Measures the association between a term and a group, indicating their dependency. |
| When to use?       | Finding terms that are characteristic for a group and are only used by a subset of other groups. | Finding terms that have higher relevance for a certain group. | Finding terms that are characteristic for a group and are seldomly used by other groups. |
| Interpretability | High Scores: indicate greater importance of the term within the group. | Positive: indicate association with the group. Negative: indicates low importance of term for the group. | High Scores: Indicate strength of association between term and group. Low Scores: indicate disassocation between term and group |

## Social Science usecase(s)
Topic Modeling and Content Analysis:

    Use Case: You want to analyze a corpus of political speeches from various politicians to identify recurring themes and topics?

    Method Application: By calculating TF-IDF, PMI, and Log Odds Ratio, you can identify significant terms and their associations within the speeches. This can help in uncovering key topics, sentiments, and political ideologies prevalent in the speeches.

Sentiment Analysis in Social Media Data:

    Use Case: You want to study public opinion on climate change using tweets from Twitter?

    Method Application: By applying TF-IDF, PMI, and Log Odds Ratio on the tweet corpus, you can identify the most relevant terms associated with discussions on climate change. This analysis can reveal sentiment polarity, key influencers, and common narratives surrounding the topic.

Comparative Analysis of Cultural Texts:

    Use Case: You want to conduct a comparative study of cultural differences in literature between two countries?

    Method Application: By utilizing TF-IDF, PMI, and Log Odds Ratio, you can compare the frequency and co-occurrence of culturally significant terms in literature from each country. This analysis can provide insights into cultural values, societal norms, and prevalent themes within the literature of each country.

## Structure
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

## Keywords
Comparative Analysis, Keyword Extraction, Word Importance, Log Odd Ratio,

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
## Input Data (DBD datasets)
- [German Federal Elections](https://search.gesis.org/research_data/ZA7721?doi=10.4232/1.13789)
- [TweetsCov19](https://data.gesis.org/tweetscov19/#dataset)
- [Call me sexist but](https://data.gesis.org/sharing/#!Detail/10.7802/2251)
- [Incels Forum Data](https://search.gesis.org/research_data/SDN-10.7802-2485?doi=10.7802/2485)
## Sample Input Data
The corpus data should look something like this:

```
{
    'Class A': "This is the liberal solution: All text is good aswell as bad. The good one has to take his own position . We are the liberal ones . Not the center nor the progressive ones.",
    'Class B': "This is the center solution: They are bad not good, if everyone remains on his own position we are all alone which is bad . We are the center ones . Not the progressive nor the liberal ones .",
    'Class C': "This is the progressive solution: Another groups position is the problem. They dont move from their position . We are the progressive ones . Not the liberal nor the center ones . "
}
```

## Sample Output
This file can be run from the Terminal with:


```
python keyword_extraction.py -help
```

This call will return the list of parameters you can specify.
Once you are familiar with the possible parameters you can run the code like this:

```
python keyword_extraction.py -method pmi -corpus /path/to/your_corpus.json
```

The method will produce a csv in the following form:

|Words | Class A | Class B | Class C|
|:---  | :---:   | :---:   | ---:|
|example | pmi_a | pmi_b | pmi_c |


Moreover, in the /config_output/ you find a json file that saved all the used parameters for the resulting table.

```
{
    "corpus": "/home/linzbasn/viewpoint-aware-language-model/Training_Variables/new_corpus.json",
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
Calling the method like this you will create the importance scores on the basis of TFIDF (Term frequency Inverse Document Frequency). This method weighs words higher when they dont appear in all documents. Meaning a word that only appears in this document is probably more important for this document than a word that is shared across all documents. With 'min_df' we specify that the words should atleast appear in 2 documents. Thus, we exclude all words that only appear in one document.

```
python keyword_extraction.py -method pmi_tfidf -corpus /path/to/your_corpus.json -less_freq_than 20
```
If you run this code you calculate the Pointwise Mutual Information for the tfidf scores of the words. This has sometimes benefits as it already takes into account how important a word is for the specific group when weighting it. Furthermore, we exclude the 20% of words that appear less often.

```
python keyword_extraction.py -method log_odds -corpus /path/to/your_corpus.json -comparison_corpus /path/to/your_comparison_corpus.json 
```
To calculate the Log Odd Ratio we need to specify a comparison corpus. This corpus should be unbiased and helps us to understand how often certain words appear under normal circumstances. With this information we can alleviate the influence of noise when we calculate the importance of our words for the corpus.


# Specifics
## Contact Details
Stephan.Linzbach@gesis.org

