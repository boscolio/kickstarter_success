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


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    goal: int = Field(..., example=50000)
    length: int = Field(..., example=30)
    description: str = Field(..., example='tabletop board game')
    category: str = Field(..., example='game')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        #return pd.DataFrame({dict(self)})
        pass

    @validator('goal')
    def goal_must_be_positive(cls, value):
        """Validate that goal is a positive number."""
        assert value > 0, f'goal == {value}, must be > 0'
        return value

    @validator('length')
    def x1_must_be_positive(cls, value):
        """Validate that length is a positive number."""
        assert value > 0, f'length == {value}, must be > 0'
        return value

@router.post('/predict')
async def predict(item: Item):
    """
    Make random baseline predictions for classification problem 🔮

    ### Request Body
    - `goal`: positive integer. US $.
    - `length`: positive integer. number of days.
    - `description`: string
    - `category`: string

    ### Response
    - `prediction`: boolean, at random
    - `predict_proba`: float between 0.5 and 1.0, 
    representing the predicted class's probability

    Replace the placeholder docstring and fake predictions with your own model.
    """

    nlp = spacy.load("en_core_web_md")

    desc_vectorized = pd.DataFrame(nlp(item.description).vector)

    X_new = desc_vectorized.T
    X_new['category'] = item.category
    X_new['goal'] = item.goal
    X_new['campaign length'] = item.length

    loaded_model = pickle.load(open('gbnlp_pickle', 'rb'))

    y_pred = loaded_model.predict(X_new)

    return {
        'prediction': y_pred
    }

    # X_new = item.to_df()
    # log.info(X_new)           
    # y_pred = random.choice([True, False])
    # y_pred_proba = random.random() / 2 + 0.5
    # return {
    #     'prediction': y_pred,
    #     'probability': y_pred_proba
    # }
