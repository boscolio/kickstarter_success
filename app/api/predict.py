import logging
import os
import random
import pickle
import datetime

from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
import numpy as np
import spacy

from datetime import datetime
from spacy.tokenizer import Tokenizer
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import randint
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder

log = logging.getLogger(__name__)
router = APIRouter()

categories = ['Academic', 'Accessories', 'Action', 'Animation', 'Anthologies',
            'Apparel', 'Apps', 'Architecture', 'Art', 'Art Books', 'Audio',
            'Bacon', 'Calendars', 'Candles', "Children's Books", 'Civic Design',
            'Classical Music', 'Comedy', 'Comic Books', 'Comics', 'Community Gardens',
            'Conceptual Art', 'Cookbooks', 'Country & Folk', 'Crafts', 'DIY',
            'DIY Electronics', 'Dance', 'Design', 'Digital Art', 'Documentary',
            'Drama', 'Drinks', 'Electronic Music', 'Events', 'Experimental',
            'Fabrication Tools', 'Faith', 'Family', 'Farms', 'Fashion', 'Festivals',
            'Fiction', 'Film & Video', 'Fine Art', 'Flight', 'Food', 'Food Trucks',
            'Footwear', 'Gadgets', 'Games', 'Gaming Hardware', 'Glass', 'Graphic Design',
            'Graphic Novels', 'Hardware', 'Hip-Hop', 'Horror', 'Illustration',
            'Immersive', 'Indie Rock', 'Installations', 'Interactive Design', 'Jazz',
            'Jewelry', 'Journalism', 'Kids', 'Live Games', 'Metal', 'Mixed Media',
            'Mobile Games', 'Music', 'Narrative Film', 'Nature', 'Nonfiction',
            'Painting', 'People', 'Performance Art', 'Performances', 'Periodicals',
            'Photo', 'Photobooks', 'Photography', 'Places', 'Playing Cards', 'Plays',
            'Poetry', 'Pop', 'Pottery', 'Print', 'Printing', 'Product Design',
            'Public Art', 'Publishing', 'Punk', 'Radio & Podcasts','Ready-to-wear',
            'Restaurants', 'Robots', 'Rock', 'Romance', 'Science Fiction', 'Sculpture',
            'Shorts', 'Small Batch', 'Software', 'Space Exploration', 'Stationery',
            'Tabletop Games', 'Technology', 'Television', 'Textiles', 'Theater',
            'Thrillers', 'Vegan', 'Video', 'Video Games', 'Wearables', 'Web',
            'Webcomics', 'Webseries', 'Woodworking', 'Workshops', 'World Music']


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    goal: int = Field(..., example=50000)
    length: int = Field(..., example=30)
    description: str = Field(..., example='tabletop board game')
    category: str = Field(..., example='Tabletop Games')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('goal')
    def goal_must_be_positive(cls, value):
        """Validate that goal is a positive number."""
        assert value > 0, f'goal == {value}, must be > 0'
        return value

    @validator('length')
    def length_must_be_positive(cls, value):
        """Validate that length is a positive number."""
        assert value > 0, f'length == {value}, must be > 0'
        return value

    @validator('category')
    def category_must_be_categories(cls, value):
        """Validate that length is a positive number."""
        assert value in categories, f'category == {value}, must be in list see above.'
        return value

@router.post('/predict')
def predict(item: Item):
    """
    Make prediction based on features listed in Request Body.

    ### Request Body
    - `goal`: positive integer. US $.
    - `length`: positive integer. Number of days.
    - `description`: string
    - `category`: string;
                 valid categories: 
                   {'Academic',
                    'Accessories',
                    'Action',
                    'Animation',
                    'Anthologies',
                    'Apparel',
                    'Apps',
                    'Architecture',
                    'Art',
                    'Art Books',
                    'Audio',
                    'Bacon',
                    'Calendars',
                    'Candles',
                    "Children's Books",
                    'Civic Design',
                    'Classical Music',
                    'Comedy',
                    'Comic Books',
                    'Comics',
                    'Community Gardens',
                    'Conceptual Art',
                    'Cookbooks',
                    'Country & Folk',
                    'Crafts',
                    'DIY',
                    'DIY Electronics',
                    'Dance',
                    'Design',
                    'Digital Art',
                    'Documentary',
                    'Drama',
                    'Drinks',
                    'Electronic Music',
                    'Events',
                    'Experimental',
                    'Fabrication Tools',
                    'Faith',
                    'Family',
                    'Farms',
                    'Fashion',
                    'Festivals',
                    'Fiction',
                    'Film & Video',
                    'Fine Art',
                    'Flight',
                    'Food',
                    'Food Trucks',
                    'Footwear',
                    'Gadgets',
                    'Games',
                    'Gaming Hardware',
                    'Glass',
                    'Graphic Design',
                    'Graphic Novels',
                    'Hardware',
                    'Hip-Hop',
                    'Horror',
                    'Illustration',
                    'Immersive',
                    'Indie Rock',
                    'Installations',
                    'Interactive Design',
                    'Jazz',
                    'Jewelry',
                    'Journalism',
                    'Kids',
                    'Live Games',
                    'Metal',
                    'Mixed Media',
                    'Mobile Games',
                    'Music',
                    'Narrative Film',
                    'Nature',
                    'Nonfiction',
                    'Painting',
                    'People',
                    'Performance Art',
                    'Performances',
                    'Periodicals',
                    'Photo',
                    'Photobooks',
                    'Photography',
                    'Places',
                    'Playing Cards',
                    'Plays',
                    'Poetry',
                    'Pop',
                    'Pottery',
                    'Print',
                    'Printing',
                    'Product Design',
                    'Public Art',
                    'Publishing',
                    'Punk',
                    'Radio & Podcasts',
                    'Ready-to-wear',
                    'Restaurants',
                    'Robots',
                    'Rock',
                    'Romance',
                    'Science Fiction',
                    'Sculpture',
                    'Shorts',
                    'Small Batch',
                    'Software',
                    'Space Exploration',
                    'Stationery',
                    'Tabletop Games',
                    'Technology',
                    'Television',
                    'Textiles',
                    'Theater',
                    'Thrillers',
                    'Vegan',
                    'Video',
                    'Video Games',
                    'Wearables',
                    'Web',
                    'Webcomics',
                    'Webseries',
                    'Woodworking',
                    'Workshops',
                    'World Music'}

    ### Response
    - `prediction`: boolean. Success or Failure.
    - `probability`: float. 

    """

    nlp = spacy.load("en_core_web_md")

    desc_vectorized = pd.DataFrame(nlp(item.description).vector)

    X_new = desc_vectorized.T
    X_new['category'] = item.category
    X_new['goal'] = item.goal
    X_new['campaign length'] = item.length

    loaded_model = pickle.load(open('gbnlp_pickle', 'rb'))

    y_pred = loaded_model.predict(X_new)
    y_prob = loaded_model.predict_proba(X_new)

    return {
        'prediction': y_pred[0],
        'probability': max(y_prob[0][0], y_prob[0][1])
    }
    