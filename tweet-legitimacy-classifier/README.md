# Tweet Legitimacy Classifier

## Instructions for how to replicate this project.

Inside this document, you will find a description of the important files related to this project. It should contain:

* README.md (This file)
* tweet_classifier.ipynb
* tweet_classifier_with_augmented_data.ipynb
* bagging_tweet_classifier.ipynb
* miscov19_final.csv
* miscov19_augmented_final.csv

## Description

We hope to delineate what each of these files are and how to run them to replicate our results.

### `miscov19_final.csv`

This file contains the Tweets in the CMU-MisCov19 dataset (Found at https://zenodo.org/record/4024154#.YfdwZlvMJhE) binned into the three labels for our classification: "legitimate", "misinformation", and "irrelevant". These labels apply in the context of the COVID-19 pandemic. Due to the Terms and Services of Twitter, we are unable to provide any more data besides the Tweet Status IDs in our datasets. This creates the additional step of scraping the text from the Twitter API using a Twitter developer account token.

### `miscov19_augmented_final.csv`

This file contains everything in miscov19_final.csv and an additional 2005 Tweets (sampled randomly from the USC dataset found here: https://github.com/echen102/COVID-19-TweetIDs). These new tweets were then classified by hand according to the same metrics described in the CMU-MisCov19 paper, and then binned into our three labels.

### Note for both datasets

Since the Twitter Terms and Service only allows the distribution of the Tweet IDs in public datasets and for research, replicating our project will require rehydrating the Tweets from their IDs. A good tutorial for how to achieve that is provided at this GitHub repository's readme: https://github.com/echen102/COVID-19-TweetIDs.

### `tweet_classifier.ipynb`

This Jupyter notebook contains the code necessary to fine-tune Digital Epidemiology's Covid-Twitter-BERT-v2 (https://huggingface.co/digitalepidemiologylab/covid-twitter-bert-v2) according to the `miscov19_final.csv` dataset. This file alludes to `miscov19_final_hydrated.csv` which is not provided by us (In compliance with Twitter TOS). This file is expected to be the same as `miscov19_final.csv` with an additional column labeled `text`. This column contains the text of each Tweet hydrated using its TweetID.

It takes a fairly long period of time to fine-tune the model. It took about 20 minutes on our hardware. Our hardware for this segment was:
* 9th Gen Intel Core i7-9750H 6 Core
* GeForce RTX 2070 Max-Q
* 40 GB DDR4-2667MHz RAM

### `tweet_classifier_with_augmented_data.ipynb`

This Jupyter notebook contains the code necessary to fine-tune Digital Epidemiology's Covid-Twitter-BERT-v2 according to the `miscov19_augmented_final.csv` dataset. This file alludes to `miscov19_augmented_final_hydrated.csv` which is not provided by us (In compliance with Twitter TOS). This file is expected to be the same as `miscov19_augmented_final.csv` with an additional column labeled `text`. This column contains the text of each Tweet hydrated using its TweetID.

Again, this model takes fairly long to train. This was about 30 minutes on our hardware (listed above).

### `bagging_tweet_classifier.ipynb`

This Jupyter notebook defines a ensemble model using four models produced by the `tweet_classifier_with_augmented_date.ipynb` notebook. It alludes to a subdirectory `./models/` which is supposed to contain four fine-tuned models by us:

* `models/first-augmented-miscov19-covid-twitter-bert-v2`
* `models/second-augmented-miscov19-covid-twitter-bert-v2`
* `models/third-augmented-miscov19-covid-twitter-bert-v2`
* `models/fourth-augmented-miscov19-covid-twitter-bert-v2`

These models were not distributed in the GitHub repository due to their size (5 GB zipped). They can be downloaded at this link:

Alternatively, you can produce the same four models using the seeds commented out in the seed cell of `tweet_classifier_with_augmented_date.ipynb`. This would alleviate the need to download such a large file, however you may need to fix some path issues by changing the `PATH` variables in this file. The drawback of this method is the amount of time it may take to train 4 models. (About 2 hours on our hardware)
