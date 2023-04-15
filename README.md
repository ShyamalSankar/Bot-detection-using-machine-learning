# Bot Detection 
In this project, we use Twitter as the setting for running experiments to differentiate bots from humans. Our project deals with this problem on 2 levels. 
1. Distinguish bots from humans at the account level (i.e. be able to distinguish bot accounts from human accounts) based on account level information (such as number of followers). 
2. Distinguish bot generated tweets from human generated tweets based on tweet level information (such as textual content of tweet). 

We also aim to investigate the value in combining different kinds of data in the same model. 
* For account level detection, we feed both image data (the profile picture of the account) and structured metadata (for example, number of followers, number of tweets etc) into the same model and compare the results with models that purely use either account metadata or account profile picture. 
* For tweet level data, we compare the model that uses both textual content of the tweet and the metadata associated with each tweet (i.e. number of likes, number of retweets etc) with models that use purely tweet level metadata or textual content. 

The findings of these experiments would be applicable in other social media contexts where bots are prevalent as well, not just for Twitter. 

**1. ETL + EDA + FE (Metadata)**
* In this repository, we perform some data preprocessing, feature selection and feature engineering based on the features obtained for each account in the webscrapping stage. We also perform exploratory data analysis on the features to attempt to see any interesting patterns in the data. We aim to get some insights into the characteristics of the metadata for the bot accounts and the human accounts respectively.

**2. ETL + EDA + FE (Text)**
* In this folder, we perform data preprocessing on tweets. Exploratory data analysis was also performed on the tweets and textual features in order to get insights into the content of bot generated tweets vs human generated tweets. 

**3. Flask**
* In this folder, we deploy the best machine learning models found during experimentation in a local flask environment. The instructions for usage are listed below.

**4. Further Analysis**

**5. Model (Image)**

**6. Model (Metadata)**
* In this folder, we trained several models to determine the optimal machine learning model that can classify an account as human or bot based on its metadata. We then pick the optimal model and save it for deployment in the flask folder.

**7. Model (Text)**
* In this folder, we trained several models to determine the optimal machine learning model that can best classify a tweet as human written or bot generated. The optimal model is picked and saved in the flask folder.

**8. Web Scraping**
* Here, we perform webscrapping using the Tweepy API to obtain the relevant features for each account.

# Flask usage instructions
Pull this repository into your local machine. Once this is done, you can install create a python virtual environment. See https://docs.python.org/3/library/venv.html for instructions to create one. Then, activate the virtual environment and then run the following command to install the required python dependencies after navigating to the flask directory:
```
pip install -r requirements.txt
```

Then, once the python packages have finished installing, you can start a flask server locally using the following command (again, ensure that you are in the flask directory).

```
python botdetection.py
```
