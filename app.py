import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from tpot import TPOTClassifier, TPOTRegressor
from streamlit_pandas_profiling import st_profile_report
import pandas_profiling
import time

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

st.header("End to end solution for your machine learning problem")
st.write("***Provides a Pipeline***")
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
            metric = st.selectbox("Choose the metric for evaluation", ["mean_absolute_error", "mean_squared_error", "r2"])
        
        X = st.session_state.data.drop(columns=[target])
        y = st.session_state.data[target]

        st.write("Setup is complete. Please wait, the best model will be shown shortly.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        stop_button = st.button("Stop Optimization")

        if stop_button:
            st.session_state.stop_optimization = True
        
        def update_progress(progress, details=""):
            progress_bar.progress(progress)
            status_text.text(f"Optimization Progress: {progress}% | {details}")

        if task == "Classification":
            automl = TPOTClassifier(verbosity=2, random_state=42, scoring=metric)
        else:
            automl = TPOTRegressor(verbosity=2, random_state=42, scoring=metric)

        try:
            for i in range(1, 101):
                if st.session_state.stop_optimization:
                    st.write("Optimization stopped.")
                    break
                simulated_details = f"{i * 101}/{10000} [04:03<43:38, 3.57 pipeline/s]"
                time.sleep(0.1)
                update_progress(i, simulated_details)
            
            if not st.session_state.stop_optimization:
                automl.fit(X, y)
                st.write("Best model:", automl.fitted_pipeline_)
                st.write("Best score:", automl.score(X, y))
                st.write("Best parameters:", automl.fitted_pipeline_.get_params())
            st.session_state.stop_optimization = False  # Reset stop flag after completion

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

        if st.button("Export Model"):
            automl.export('best_model.py')
            st.success("Model exported successfully.")

        progress_bar.progress(100)
        status_text.text("Progress: 100%")
        
    else:
        st.warning("No data available. Please upload a dataset first.")

def download_step():
    st.write("Download options will be provided here.")
    st.write("Thank you for using the application!")

# Show the current step
show_step(st.session_state.step)

# Add "Back" and "Next" buttons
col1, col2 = st.columns([1, 1])
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
