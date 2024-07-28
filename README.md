**OptiML**

This is a Streamlit web app that provides end-to-end solutions by providing a pipeline in Python based on the user's requirements. This app includes exploratory data analysis of the data, choosing your target variable, task type(regression, classification), metric, no of epochs, and even cross-validation. This is a customizable yet no-code app that creates a perfect pipeline.
## Tech Stack

**Client:** Streamlit

**Server:** Python , TPOT

**Libraries and Tools:**
1) pandas- For data manipulation
2) tpot - For automated machine learning
3) streamlit_pandas_profiling -For data profiling
4) pandas_profiling -For generating profile reports
5) pickle -For model serialization
6) sklearn- For evaluation metrics



## Deployment

To deploy this project locally run

For Windows

```bash
git clone https://github.com/username/OptiML.git
```
```bash
cd OptiML
```
Ensure you have conda installed
```bash
conda env create -f environment.yml
```
Replace your_env_name to any name you want to name the environment
```bash
conda activate your_env_name
```
Install Dependencies

```bash
pip install -r requirements.txt
```
Now run the app

```bash
streamlit run app.py
```

## Authors

- [@Vardhani30](https://github.com/Vardhani30)


## Acknowledgements

 - [Nicholas Renotte](https://github.com/nicknochnack)
 
