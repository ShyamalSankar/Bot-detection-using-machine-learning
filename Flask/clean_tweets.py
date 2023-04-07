import pandas as pd
import tweepy
import re
import numpy as np
import os
import demoji
import emoji
import string
from bs4 import BeautifulSoup


#to deal with html characters
CLEANR = re.compile('<.*?>') 
#replace the new line characters
def process_text(text):
    txt_lst = text.split()
    
    #A helper function to process emojis
    #Emojis are left in in order to 
    def process_emoji(emo):
        try:
            decoded = emoji.demojize(emo)
            decoded = decoded.replace(":", "")
            return decoded
        except UnicodeDecodeError:
            #if unable to decode emoji, just keep a place holder for it
            return "__emoji__"
    
    #store all emojis as the decoded form
    txt_lst = [process_emoji(x) if emoji.is_emoji(x) else x for x in txt_lst]
    
    #process all tagged accounts
    def process_tagged_accounts(account):
        #replace all tagged accounts with __user_mention__
        if account.startswith("@") and len(account) > 1:
            return "__user_mention__"
        return account
    
    #replace hashtags with place holders
    def process_hashtags(text):
        if text.startswith("#"):
            return "__hashtag__"
        return text
    
    
    #apply the functions above
    txt_lst = [process_tagged_accounts(x) for x in txt_lst]
    txt_lst = [process_hashtags(x) for x in txt_lst]
    
    #next, we process the urls
    def process_urls_html(text):
        pattern = r'(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)'
        text = re.sub(pattern, "", text)
        text = re.sub(CLEANR, "", text)
        return text
    
    text = " ".join(txt_lst)
    
    final_text = process_urls_html(text)
    return final_text

"""
def preprocess(text):
    #remove emojis and store decoded emojis in list
    all_emojis = emoji_code_text(text)
    text = ''.join(x for x in text if not emoji.is_emoji(x))
    #remove url
    pattern = r'(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)' 
    text = re.sub(pattern, '', text)
    text = text.strip()
    text = text.lower()
    #text = word_tokenize(text)
    text = text.split(' ')
    text = [w for w in text if not w in stopwords]
    final_text = []
    for word in text:
        #check useraccount
        if word.startswith("@"):
            final_text.append("__user_mention__")
        #check hashtag
        elif word.startswith("#"):
            final_text.append("__hashtag__")
        else:
            final_text.append(lemmatizer.lemmatize(word))
    final_text = final_text + all_emojis
    return " ".join(word for word in final_text)

def emoji_code_text(text):
    all_emojis = ''.join(x for x in text if emoji.is_emoji(x))
    #emoji_dict = dict()
    emoji_word_list = []
    try:
        #For each emoji, decode it and add the decoding into the list
        for each_emoji in all_emojis:
            decoded = emoji.demojize(each_emoji)
            decoded = decoded.replace(':', '')
            emoji_word_list.append(decoded)
    except UnicodeDecodeError:
        emoji_word_list = []
        #df["emoji"][word] = dict()
        #df["emoji_text"][word] = []
    return emoji_word_list

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def remove_nonalphanum(text):
    return re.sub("[^a-z0-9]","", text)
"""

