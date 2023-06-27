#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from PIL import Image
import pickle


# In[14]:


# Load the dataset
df = pd.read_csv("Telco_Customer.csv")


# In[17]:


# Separate the features and target variable
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Encode the categorical variables
categorical_vars = X.select_dtypes(include="object").columns.tolist()
encoders = {}
for var in categorical_vars:
    encoder = LabelEncoder()
    X[var] = encoder.fit_transform(X[var])
    encoders[var] = encoder

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the AdaBoost classifier
adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)

# Save the trained model using pickle
model_filename = "adaboost_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(adaboost, file)

# Define prediction function using the loaded model
def predict_churn(*data):
    # Load the saved model
    with open(model_filename, "rb") as file:
        loaded_model = pickle.load(file)

    # Encode the inputs
    encoded_data = []
    for i, var in enumerate(categorical_vars):
        encoder = encoders[var]
        encoded_value = encoder.transform([data[i]])[0]
        encoded_data.append(encoded_value)

    # Make predictions on the encoded data
    encoded_df = pd.DataFrame([encoded_data], columns=X.columns)
    prediction = loaded_model.predict(encoded_df)

    # Save inputs and output to CSV
    input_data = pd.DataFrame([data], columns=X.columns)
    input_data["Churn Prediction"] = prediction[0]
    input_data.to_csv("history.csv", index=False)

    return "Churn Prediction: " + prediction[0]

# Create the dropdown choices using raw data
dropdown_choices = {}
for var in categorical_vars:
    dropdown_choices[var] = list(df[var].unique())
    

# Create the input interfaces using Gradio
input_interfaces = [gr.inputs.Dropdown(choices=dropdown_choices[col], label=col) for col in X.columns]
output_interface = gr.outputs.Textbox()

# Create the Gradio interface
iface = gr.Interface(fn=predict_churn, inputs=input_interfaces, outputs=output_interface, title = "Customer Churn Prediction App")


# Run the Gradio interface
iface.launch(share =True)


# In[ ]:




