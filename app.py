import streamlit as st
import pandas as pd
import numpy as np
import pickle

header = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

with header:
  st.title("Breast Cancer Detection")

with dataset:
  st.header("Using the Breast Cancer Wisconsin Data Set")
  sample_data = pd.read_csv("/content/drive/MyDrive/AIML PROJECT - BREAST CANCER DETECTION/data.csv")
  st.write(sample_data.head())

with features:
  st.header("Key features of the project:")
  st.markdown("* **First Feature**: The aim of this analysis is to use Logistic Regression to classify the data into two classes of diagnosis— Malignant & Benign ")
  st.markdown("* **Second feature**: The Wisconsin Breast Cancer (Diagnostic) dataset has been created by computing from a digitized image of a fine needle aspirate (FNA) of a breast mass.")
  st.markdown("* **Third feature**: Some attributes include - a) radius (mean of distances from center to points on the perimeter) b) texture (standard deviation of gray-scale values) c) perimeter d)area etc for each cell nucleus")

with modelTraining:
  st.header("Running the model with the sample data!")

  filename = 'finalized_model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))

  sel_col, disp_col = st.columns(2)

  input_feature = sel_col.text_input("Enter sample data:")

  try:
      res = tuple(map(float, input_feature.split(',')))
  
  except Exception:
        st.error("Please make sure that you enter valid input!")
        st.stop()

#include any sample data from the data.csv file to cross check results
input_data = res

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)


st.subheader("RESULT:")
if (prediction[0] == 0):
  st.text('The Breast cancer is Malignant')

else:
  st.text('The Breast Cancer is Benign')





