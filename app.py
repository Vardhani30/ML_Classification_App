from operator import index
import streamlit as st
import plotly.express as px
from pycaret.classification import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

