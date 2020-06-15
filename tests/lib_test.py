# -*- coding: UTF-8 -*-

# Import from standard library
import os
import datatools
import pandas as pd
# Import from our lib
from datatools.lib import clean_data
from datatools.text import preprocess_text_dataframe

import pytest


def test_clean_data():
	datapath = os.path.dirname(os.path.abspath(datatools.__file__)) + '/data'
	df = pd.read_csv('{}/data.csv.gz'.format(datapath))
	first_cols = ['id', 'civility', 'birthdate', 'city', 'postal_code', 'vote_1']
	assert list(df.columns)[:6] == first_cols
	assert df.shape == (999, 142)
	out = clean_data(df)
	assert out.shape == (985, 119)

def test_preprocess_text_dataframe():
	datapath = os.path.dirname(os.path.abspath(datatools.__file__)) + '/data'
	df = pd.read_csv('{}/data.csv.gz'.format(datapath))
	df_processed = preprocess_text_dataframe(df, 'city')
	assert df_processed.shape == df.shape

