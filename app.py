import streamlit as st
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from tpot import TPOTClassifier, TPOTRegressor
from streamlit_pandas_profiling import st_profile_report
import time

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


if 'data' not in st.session_state:
    st.session_state.data = None


with st.sidebar:
    st.image("https://w7.pngwing.com/pngs/292/543/png-transparent-robot-humanoid-robot-military-robot-artificial-intelligence-robotics-electronics-desktop-wallpaper-robot.png")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")


if choice == "Upload":
    st.write("Upload your data here.")
    file = st.file_uploader("Upload your dataset")
    if file:
        st.session_state.data = pd.read_csv(file, index_col=None)
        st.success("Data uploaded successfully")

elif choice == "Profiling":
    st.write("Profiling of data will be displayed here.")
    if st.session_state.data is not None:
        profile_df = st.session_state.data.profile_report()
        st_profile_report(profile_df)
    else:
        st.warning("No data available. Please upload a dataset first.")

elif choice == "Modelling":
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

        progress_bar=st.progress(0)
        status_text=st.empty()
        stop_button=st.button("Stop Setting Up")

        if stop_button:
            st.session_state.stop_optimization = True
        
        def update_progress(progress):
            progress_bar.progress(progress)
            status_text.text(f"Progress: {progress}%")

    
        if task == "Classification":
            automl = TPOTClassifier(verbosity=2, random_state=42, scoring=metric)
        else:
            automl = TPOTRegressor(verbosity=2, random_state=42, scoring=metric)

        try:
            for i in range(1, 101):
                if st.session_state.stop_optimization:
                    st.write("Optimization stopped.")
                    break
                time.sleep(0.1)  # Simulate computation
                update_progress(i)
            if not st.session_state.stop_optimization:
                automl.fit(X, y)
                st.write("Best model:", automl.fitted_pipeline_)
                st.write("Best score:", automl.score(X, y))
                st.write("Best parameters:", automl.fitted_pipeline_.get_params())
            st.session_state.stop_optimization = False  # Reset stop flag after completion

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()
        

        st.write("Best model:", automl.fitted_pipeline_)
        st.write("Best score:", automl.score(X, y))
        st.write("Best parameters:", automl.fitted_pipeline_.get_params())

        if st.button("Export Model"):
                automl.export('best_model.py')
                st.success("Model exported successfully.")

        progress_bar.progress(100)
        status_text.text("Progress: 100%")
    else:
        st.warning("No data available. Please upload a dataset first.")

elif choice == "Download":
    st.write("Download options will be provided here.")
