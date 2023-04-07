# Imports
import os
import tweepy
from datetime import datetime
import re
import time
import numpy as np
import pandas as pd
import pickle
import flask
from other import ExperimentalTransformer
import shap
import transformers
from transformers import BertModel, BertTokenizer, pipeline
from clean_tweets import process_text


app = flask.Flask(__name__)

xgb_model = pickle.load(open('XGBoost.sav','rb'))
shap_explainer = pickle.load(open('xgb_shap.sav','rb'))

#Load the pretrained BERT model
tokenizer = BertTokenizer.from_pretrained("bert-base-cased", padding = True)
#load the pretrained model
bert_model = BertModel.from_pretrained("bert-base-cased")
#create a pipeline in which the tweets get converted to bert features
nlp = pipeline("feature-extraction", tokenizer=tokenizer, model=bert_model)

#load the model that is used to predict whether a tweet is made by a human or a bot
tweet_predictor = pickle.load(open('Bert_model.sav','rb'))

def bot_likelihood(prob):
    if prob < 20:
        return '<span class="info">Not a bot</span>'
    elif prob < 35:
        return '<span class="info">Likely not a bot</span>'
    elif prob < 50:
        return '<span class="info">Probably not a bot</span>'
    elif prob < 60:
        return '<span class="warning">Maybe a bot</span>'
    elif prob < 80:
        return '<span class="warning">Likely a bot</span>'
    else:
        return '<span class="danger">Bot</span>'
    
def bot_proba(twitter_handle):
    '''
    Takes in a twitter handle and provides probabily of whether or not the user is a bot
    Required: trained classification model (XGBoost) and user account-level info from get_user_features
    '''
    user_features = get_user_features(twitter_handle)
    user = np.matrix(user_features)
    print(user)
    df = pd.DataFrame(user, columns=["protected", "verified", "location", "followers_count", "following_count", "tweet_count",
                                     "listed_count", "has_profile_image", "un_no_of_char",
                                     "un_special_char", "name_no_of_char", "name_special_char","des_no_of_usertags",
                                     "des_no_of_hashtags", "des_external_links", "has_description", "has_url"])
    print(df)
    if user_features == 'User not found':
        return 'User not found'
    else:
        user = np.matrix(user_features)
        proba = np.round(xgb_model.predict_proba(df)[:, 1][0]*100, 2)
        print(proba)
        return proba

@app.route('/')
def homepage():
    return flask.render_template('landing.html')

@app.route('/byhandle')
def byhandle():
    return flask.render_template('handle.html')

@app.route('/bytweet')
def bytweet():
    return flask.render_template('tweet.html')

@app.route('/back')
def back():
    return flask.render_template('landing.html')

@app.route('/predicthandle', methods=['GET', 'POST'])
def make_prediction_handle():
    handle = flask.request.form['handle']
    print(handle)

    # make predictions with model from twitter_funcs
    user_lookup_message = f'Prediction for @{handle}'
    user_features = get_user_features(handle)
    #print(get_user_features(handle))
    shap_plot = ""
    text = ""
    print(user_features)
    if get_user_features(handle) == 'User not found':
        prediction = [f'User @{handle} not found', '']

    else:
        text = "Shap Explainer:"
        prediction = [bot_likelihood(bot_proba(handle)),
                      f'Probability of being a bot: {bot_proba(handle)}%']
        user = np.matrix(user_features)

        user_features = get_user_features(handle)
        # user = np.matrix(user_features)
        explainer = shap_explainer
        # shap_values = explainer.shap_values(user)
        user = pd.DataFrame(user_features, index = ["protected", "verified", "location", "followers_count", "following_count", "tweet_count",
                                                     "listed_count", "has_profile_image", "un_no_of_char",
                                                     "un_special_char", "name_no_of_char", "name_special_char","des_no_of_usertags",
                                                     "des_no_of_hashtags", "des_external_links", "has_description", "has_url"]).T

        shap_values = explainer(user)
        shap_values = shap_values[...,1]
        def _force_plot_html(explainer, user_features):
            force_plot = shap.plots.force(shap_values)
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            return shap_html
        shap_plot = _force_plot_html(explainer, user_features)

    return flask.render_template('handle.html', prediction=prediction[0], probability=prediction[1], user_lookup_message=user_lookup_message, text = text, shap_plots = shap_plot)

@app.route('/predicttweet', methods=['GET', 'POST'])
def make_prediction_tweet():
    tweet = flask.request.form['handle']
    #preprocess the incoming tweet
    processed_tweet = process_text(tweet)
    #convert the tweet to BERT encoding
    bert_features = np.array(nlp(processed_tweet))
    #getting the mean representation of the tweet
    bert_features = bert_features.reshape((bert_features.shape[1], bert_features.shape[2])).mean(axis = 0)
    #extracting the probability from the predict proba method of the model
    probability = tweet_predictor.predict_proba(bert_features.reshape(1,-1))[0][1]
    percentage = round(probability * 100, 3)
    

    text = "Prediction for Tweet:"
    #user_lookup_message = f'Prediction for Tweet {tweet}'
    prediction = [bot_likelihood(percentage), f'Probability that this tweet is by a bot: {round(probability*100, 2)}%']
    #final_statement = f"The chance that this tweet: \n '{tweet}' \n was made by a bot is {percentage}%"
    
    return flask.render_template('tweet.html', prediction=prediction[0], probability=prediction[1], user_lookup_message=tweet, text=text)
    #flask.render_template('tweet.html', text = final_statement)

bearer_token = 'AAAAAAAAAAAAAAAAAAAAADuChwEAAAAAyd5NyoPPZfk%2FiBwmc2mC9me33RA%3DTFH93ScdBzcU6OHVLLsTDHKLW599NhhPoEBTPi0KFWdAEbmFth'
client = tweepy.Client(bearer_token=bearer_token)

def get_user_features(screen_name):
    '''
    Input: a Twitter handle (screen_name)
    Returns: a list of account-level information used to make a prediction 
            whether the user is a bot or not
    '''
    
    try:
        # Get user information from screen name
        user = client.get_user(username=screen_name,
                                user_fields=["created_at",
                                             "description",
                                             "entities",
                                             "id",
                                             "location",
                                             "name",
                                             "profile_image_url",
                                             "protected",
                                             "public_metrics",
                                             "url",
                                             "username",
                                             "verified", 
                                             "withheld"])
        data = user.data
        
        # account features to return for predicton
        protected = int(data["protected"] == True)
        verified = int(data["verified"] == True)
        location = int(data["location"] == True)
        followers_count = data["public_metrics"]['followers_count']
        following_count = data["public_metrics"]['following_count']
        listed_count = data["public_metrics"]['listed_count']
        tweet_count = data["public_metrics"]['tweet_count']
        name = data["name"]
        username = data["username"]
        description = data["description"]
        url = data["url"]
        has_description = int(description != None) # int(data["description"]!="")
        has_url = int(url != None)
        profile_image_url = data["profile_image_url"]
        
        # manufactured features
        user_tags = r'\B@\w*[a-zA-Z]*\w*'
        hashtags = r'\B#\w*[a-zA-Z]+\w*'
        links = r'(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)'
        special_char = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        un_no_of_char = len(username)
        un_special_char = int(special_char.search(username)!= None == True)
        # un_uppercase = int(bool(re.match(r'\w*[A-Z]\w*', username)) == True)
        name_no_of_char = len(name)
        name_special_char = int(special_char.search(name)!= None == True)
        # name_uppercase = int(bool(re.match(r'\w*[A-Z]\w*', name)) == True)
        des_no_of_usertags = len(re.findall(user_tags, description))
        des_no_of_hashtags = len(re.findall(hashtags, description))
        des_external_links = int(re.findall(links, description) != [] == True)
        account_age_in_days = (datetime.now() - data['created_at'].replace(tzinfo=None)).days
        has_profile_image = int(profile_image_url.find("/default_profile_normal.png") == -1 if type(profile_image_url) == str else False)

        # organizing list to be returned
        account_features = [protected, verified, location, followers_count, following_count, tweet_count, listed_count, has_profile_image,
                            un_no_of_char, un_special_char, name_no_of_char, name_special_char,
                            des_no_of_usertags, des_no_of_hashtags, des_external_links, has_description, has_url]

    except: #Exception as e:
        return'User not found'
    
    return account_features if len(account_features) == 17 else f'User not found'

# for local dev
if __name__ == '__main__':
    #debug = true just updates the thing everytime you save it
    app.run(port = 5001) #debug=True
