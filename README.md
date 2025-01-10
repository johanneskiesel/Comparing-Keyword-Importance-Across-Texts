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
The method consists of one file [keyword_extraction.py](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/blob/main/keyword_extraction.py).
The data used for the demo run is saved in the [data/](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/tree/main/data/) folder.
Once the method is finished the method will create the following folder structure and output.
In the [ouptut/](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/tree/main/output/) folder you find the csv with the word importance scores.
In [output_config/](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/tree/main/output_config/) you find a json specify all the parameters used to produce the csv.

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



## Environment Setup
Install Python v>=3.9 (preferably through Anaconda).

Download the repository with or directly copy the raw code from [keyword_extraction.py](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/blob/main/keyword_extraction.py), and requirements.txt
```
git clone https://git.gesis.org/bda/keyword_extraction.git
```



## Installing Dependencies
- Installing all the packages and libraries with specific versions required to run this method

  ```
  pip install -r requirements.txt
  ```

- The user should be able to reuse the method following the information provided


## Input Data
The method handles digital behavior data, including social media posts, comments, search queries, clickstream text (e.g., website titles), forum threads, and open-text survey responses.

## Sample Input Data
The corpus data used in the script is stored in JSON format at [data/default_corpus.json](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/blob/main/data/default_corpus.json) and looks something like this: 

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
progressive |	0.24816330799414105	| 0.24816330799414105	|1.2392023539955106
ones	| 0.636647135255376 |	0.636647135255376 |	0.6276861812567451
position |	0.24816330799414105 |	0.24816330799414105 |	1.2392023539955106
solution |	0.636647135255376 |	0.636647135255376	| 0.6276861812567451
center	| 0.20851385530561406	| 1.208513855305614	| 0.19955290130698336
liberal	| 1.208513855305614	| 0.20851385530561406	| 0.19955290130698336


Moreover, in the [output_config/](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/tree/main/output_config) you find a json file that saved all the used parameters for the resulting table.

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
You can configure the parameters in the [config.json](https://github.com/Stephan-Linzbach/Comparing-Keyword-Importance-Across-Texts/blob/main/config.json) file and run the script:
```
python keyword_extraction.py
```
Alternatively, you can set the parameters directly in the command line when running the script.

### Example Commands and Parameters
__1. Pointwise Mutual Information (PMI)__
   Calculate PMI for words that appear in more than 80% of the documents:
```
python keyword_extraction.py --config False --method pmi --corpus /path/to/your_corpus.json --more_freq_than 80
```
Words that occur frequently across documents are prioritized.

__2. TF-IDF (Term Frequency-Inverse Document Frequency)__
Compute importance scores based on TF-IDF, excluding words that appear in fewer than two documents:
```
python keyword_extraction.py --config False --method tfidf --corpus /path/to/your_corpus.json --min_df 2
```
TF-IDF highlights words that are unique to specific documents compared to those shared across all documents. With 'min_df' we specify that the words should at least appear in 2 documents. Thus, we exclude all words that only appear in one document.

__3. PMI with TF-IDF__
Combine PMI with TF-IDF scores, excluding the least frequent 20% of words:
```
python keyword_extraction.py --config False --method pmi_tfidf --corpus /path/to/your_corpus.json --less_freq_than 20
```
This approach accounts for both document-specific importance and overall word weighting.

__4. Log Odds Ratio__
Compute Log Odds Ratio using a comparison corpus to identify word importance:
```
python keyword_extraction.py --config False --method log_odds --corpus /path/to/your_corpus.json --comparison_corpus /path/to/your_comparison_corpus.json 
```
A comparison corpus is required to determine word frequencies under "normal" circumstances, reducing noise and highlighting significant terms.




## Contact Details
Stephan.Linzbach@gesis.org

