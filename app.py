import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from tpot import TPOTClassifier, TPOTRegressor
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling
import time
import pickle 
from sklearn.metrics import SCORERS

# Set up the top bar with a transparent background
st.markdown("""
    <style>
    .top-bar {
        background-color: rgba(0, 0, 0, 0);  /* Transparent background */
        color: #ffffff;  /* White text color */
        padding: 10px;
        border-bottom: 2px solid #444444;  /* Darker border for contrast */
        display: flex;
        align-items: center;
    }
    .top-bar img {
        vertical-align: middle;
        background-color: rgba(0, 0, 0, 0);  /* Transparent background for image */
        margin-right: 10px;
    }
    .top-bar h1 {
        display: inline;
        margin-left: 10px;
    }
    </style>
    <div class="top-bar">
        <img src="https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png" width="200" />
        <h1>OptiML</h1>
    </div>
""", unsafe_allow_html=True)

st.header("End to end solution to your machine learning problem")
# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'stop_optimization' not in st.session_state:
    st.session_state.stop_optimization = False

# Define the steps
steps = ["Upload", "Profiling", "Modelling", "Download"]

def show_step(step):
    if step == 0:
        upload_step()
    elif step == 1:
        profiling_step()
    elif step == 2:
        modelling_step()
    elif step == 3:
        download_step()

def upload_step():
    file = st.file_uploader("Upload your CSV dataset")
    if file:
        st.session_state.data = pd.read_csv(file, index_col=None)
        st.write(st.session_state.data.head(10))
        st.success("Data uploaded successfully")

def profiling_step():
    st.write("Profiling of data will be displayed here.")
    if st.session_state.data is not None:
        profile_df = st.session_state.data.profile_report()
        st_profile_report(profile_df)
    else:
        st.warning("No data available. Please upload a dataset first.")

def modelling_step():
    st.write("Modeling options will be displayed here.")
    if st.session_state.data is not None:
        columns = st.session_state.data.columns.tolist()
        target = st.selectbox("Select your target column", columns)
        task = st.selectbox("Select task type", ["Classification", "Regression"])
        
        if task == "Classification":
            metric = st.selectbox("Choose the metric for evaluation", ["accuracy", "f1", "precision", "recall", "roc_auc"])
        else:
            metric = st.selectbox("Choose the metric for evaluation", ["neg_mean_absolute_error", "neg_mean_squared_error", "r2"])

        generations = st.slider("Select the number of generations", min_value=5, max_value=1000, value=10)
        cv_splits = st.slider("Select the number of cross-validation splits", min_value=3, max_value=10, value=3)
        
        X = st.session_state.data.drop(columns=[target])
        y = st.session_state.data[target]

        if st.button("Run Modelling"):
            st.write("Setup is complete. Please wait, the best model will be shown shortly.")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            st.session_state.stop_optimization = False

            def update_progress(progress, details=""):
                progress_bar.progress(progress)
                status_text.text(f"Optimization Progress: {progress}% | {details}")

            if task == "Classification":
                automl = TPOTClassifier(verbosity=2, random_state=42, scoring=metric,n_jobs=-1,cv=cv_splits,generations=generations)
            else:
                automl = TPOTRegressor(verbosity=2, random_state=42, scoring=metric,n_jobs=-1,cv=cv_splits,generations=generations)

            try:
                for i in range(1, 101):
                    if st.session_state.stop_optimization:
                        st.write("Optimization stopped.")
                        break
                    time.sleep(0.1)
                    update_progress(i)
                
                if not st.session_state.stop_optimization:
                    automl.fit(X, y)
                    st.session_state.best_model = automl.fitted_pipeline_
                    st.session_state.best_score = automl.score(X, y)
                    st.session_state.best_params = automl.fitted_pipeline_.get_params()
                    
                    automl.export('best_model.py')

                    # Save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(automl.fitted_pipeline_, f)
                        
                st.session_state.stop_optimization = False  # Reset stop flag after completion

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

            progress_bar.progress(100)
            status_text.text("Progress: 100%")
            
    else:
        st.warning("No data available. Please upload a dataset first.")

# To display the best model information in a separate block
if 'best_model' in st.session_state:
    st.write("Best Model Information")
    st.write("Best model:", st.session_state.best_model)
    st.write("Best score:", st.session_state.best_score)
    st.write("Best parameters:", st.session_state.best_params)

    with open('best_model.py', 'r') as file:
        st.code(file.read(), language='python')


def download_step():
    st.write("Thank you for using the application!")
    if 'best_model.pkl' in st.session_state:
        with open('best_model.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name="best_model.pkl")


show_step(st.session_state.step)

col1, col2 = st.columns([1, 0.1])
with col1:
    if st.session_state.step > 0:
        if st.button("Back"):
            st.session_state.step -= 1
            st.experimental_rerun()

with col2:
    if st.session_state.step < len(steps) - 1:
        if st.button("Next"):
            st.session_state.step += 1
            st.experimental_rerun()
