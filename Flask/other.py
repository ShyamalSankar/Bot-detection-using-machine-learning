from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

class ExperimentalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_variables = ['followers_count', 'following_count', 'tweet_count', 'un_no_of_char','name_no_of_char',
                                            'des_no_of_usertags', 'des_no_of_hashtags', 'account_age_in_days']):
        self.encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)
        self.scaler = StandardScaler()
        #specified numeric variables, by default it is the above
        self.numeric_variables = numeric_variables
        self.columns = []
    
    #The fit function that will be called when this custom transformer is fit
    def fit(self, X, y = None):
        #fit the one hot encoder to the year
        #self.encoder.fit(X[['year']])
        #fit the scaler on the numeric variables
        self.scaler.fit(X[self.numeric_variables])
        return self
    
    #The transform function that will be called
    def transform(self, X, y = None):
        #to avoid changing the original dataset
        X_ = X.copy()
        #transforming the numeric variables according to the fitted scaler
        X_[self.numeric_variables] = self.scaler.transform(X_[self.numeric_variables])
        self.columns = X_.columns
        return X_

