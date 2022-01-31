# ProjectX-2021
This is the GitHub repository for Cornell Data Science's 2021 ProjectX team. Members: Alexander Wang, Jerry Sun, Kevin Zhou, Kaitlyn Chen, Edward Gu, Melinda Fang.

## Instructions for how to replicate our results:
In the following headings, we describe how each of our subproject experiments can be replicated.

## Data Collection
The USC data source can be found here: https://github.com/echen102/COVID-19-TweetIDs 

The CMU data source can be downloaded here: https://zenodo.org/record/4024154#.YVTHH5rMJPZ

The CMU data source can be hydrated manually by running the `data/MiscovData.ipynb` file. The only features added are the text from the tweets. A note on hydration is that the tweets are hydrated if they still exist at the time of hydration (i.e. if a tweet has been deleted then it won't show up anymore). 

The USC data source is exhaustive and contains over ~2 billion tweets. The `data/CovMasterSet.ipynb` notebook samples tweet ids from the collection and compiles them into a CSV file. This file's hydration can then be completed through the use of the [Hydrator](https://github.com/DocNow/hydrator) application which can be downloaded locally and then used to populate features according to the documentation.

## ClaimBuster
In this section you can find code to classify claims and non-claims.
- Bi-Directional LSTM implementation of ClaimBuster<a href="https://colab.research.google.com/drive/1wNmkwNExu641akHIvOtkOVTMEbDrdmdo"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- <a href="https://github.com/idirlab/claimspotter">link to original ClaimBuster repository</a>

## Tweet Legitimacy Classifier
Check the README.md inside of the tweet-legitimacy-classifier directory.

## Virality Analysis
The pre-processing conducted can be run here: https://colab.research.google.com/drive/1AcasEIEHxz07N9FJ5EUmLitTqrVOlKhk?usp=sharing 

Ensure that when running any of the drive/file path commands that your local directories match either the given paths or that you alter the file path to match the structure of your local system. 

### Some notes on running the pre-processing notebook:
* The first few code blocks of the preprocessing notebook in Colab include data scraping through the Twitter API which takes several hours to complete. 
* Running the processed text through BERT to obtain the word vector embeddings also takes several hours. There is also some difficulty in running the data through the BERT model causing the kernel to crash fairly frequently. There is a tedious work around that involves running the code cell that initializes and creates the datasets as well as the loop mechanism cell that feeds the data through the BERT forward pass manually repeatededly (our attempts at automating this process were not successful as the kernel would crash). Follow the comments in those cells to run that process properly.

The classification and regression models can be found and run with ease here: https://colab.research.google.com/drive/1nArfr4hv7V-is2LYgz4PLyuqHcDkKOIc?usp=sharing

## Full Pipeline Analysis
The curation of data and analysis of results can be found in this notebook: https://colab.research.google.com/drive/1uF7UoZY55Ybmh0TlvNPAbW4V_jJK0uFa?usp=sharing

Note that this notebook does not include code that runs the data through the entire pipeline. It only includes the creation of the dataset as well as any analysis or derived insights.
