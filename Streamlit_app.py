import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

ROOT_DIR = os.getcwd()
SAVED_DIR_PATH = "saved_models"
# MODEL_TRAINER_FOLDER="model_trainer"
# MODEL_FILE_DIR ="model"
MODEL_FILE_NAME = "xgb_model.joblib"
# TRANSFORMER_FILE_DIR="transformer"
TRANSFORMER_FILE_NAME="transformer.joblib"
# TARGET_ENCODER_FILE_DIR="target_encoder"
# TARGET_ENCODER_FILE_NAME="target_encoder.pkl"

MODEL_DIR = os.path.join(ROOT_DIR, SAVED_DIR_PATH,MODEL_FILE_NAME)
# print("MODEL_PATH:-",MODEL_DIR)

TRANSFORMER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,TRANSFORMER_FILE_NAME)
# print("TRANSFORMER_PATH:-",TRANSFORMER_DIR)

# TARGET_ENCODER_DIR= os.path.join(ROOT_DIR, SAVED_DIR_PATH,SAVED_ZERO_FILE,TARGET_ENCODER_FILE_DIR,TARGET_ENCODER_FILE_NAME)
# print("TARGET_ENCODER_PATH:-",TARGET_ENCODER_DIR)

# Load the Model.pkl, Transformer.pkl and Target.pkl
# model=pickle.load(open(MODEL_DIR,"rb"))
model=joblib.load(MODEL_DIR)
# print(model)
# transfomer=pickle.load(open(TRANSFORMER_DIR,"rb"))
transfomer=joblib.load(TRANSFORMER_DIR)

# print(transfomer)

#Read dataset
df = pd.read_csv("https://github.com/Milind-Shende/census/raw/main/census.csv")


st.set_page_config(layout="wide") 

# Set background image using CSS
st.markdown(
    """
    <style>
    body {
        background-image: url("path_to_your_background_image.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# About page
def about_page():
    

    st.title('IncomeWise: Annual Income Prediction')

    st.write("**Project Overview:** :notebook:")
    st.write("The primary objective of this project is to develop a machine learning model that predicts whether an individual's annual income exceeds 50000 based on census data. This dataset, commonly referred to as the \"Census Income\" dataset, serves as the foundation for the predictive model. The project falls within the domain of binary classification, where the goal is to categorize individuals into one of two income groups: those with incomes greater than 50000 and those with incomes less than or equal to 50,000.")
    
    # st.write("**Dataset Description:**")
    # st.write("The \"Census Income\" dataset comprises a diverse range of attributes, including age, education level, occupation, marital status, and more, which are utilized as features for the prediction task. Each individual in the dataset is associated with a binary income label, making it suitable for supervised learning. The dataset serves as a valuable resource for understanding the factors that contribute to income disparities and for building predictive models aimed at addressing this critical socioeconomic challenge.")

    st.write("**Machine Learning Approach:** :bar_chart:")
    st.write("To accomplish this prediction task, we employ a machine learning model, which is trained on a subset of the dataset. The model learns to discern patterns and relationships within the data that correlate with income levels. Once trained, it can then be used to make predictions on new, unseen data, providing insights into an individual's likelihood of earning an income above or below the 50,000 threshold.")
    
    st.write('**Project Outcome:**:blue_book:')
    st.write('The anticipated outcome of this project is a robust and accurate predictive model that can classify individuals into income groups, thereby assisting in identifying those at risk of lower incomes. This information can be leveraged to inform initiatives that aim to improve economic opportunities and equity.By combining the power of data analysis and machine learning, this project endeavors to provide valuable insights into income prediction and contribute to our understanding of socioeconomic factors influencing individual incomes.')

    st.title("Dataset Source")
    st.write("In our dataset we have 14 columns 48842 Rows which reflect various attributes of the Person Salary or Income. The target column is Salary , This dataset contains information related to individuals and aims to predict whether a person's annual salary is greater than 50,000 or less than or equal to 50,000 based on various attributes. It includes features such as age, working class, education, marital status, occupation, gender, capital gains, capital losses, hours worked per week, and country of residence.")    
    st.write(" :link: Kaggle link :- https://www.kaggle.com/datasets/overload10/adult-census-dataset?resource=download")
    st.write(" :link: UCI Repository :- https://archive.ics.uci.edu/dataset/2/adult")

def visualization_page():

    # Create a simple bar chart using Plotly Express
    fig1 = px.pie(df, names='education', title='Pie Chart Of Education & Salary', color='education', color_discrete_sequence=px.colors.qualitative.Plotly)

    fig2 = px.pie(df, names='workclass', title='Pie Chart Of Workclass & Salary', color='workclass', color_discrete_sequence=px.colors.qualitative.Plotly)

    # Display the chart using st.plotly_chart
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(df, x='occupation', y='salary', title='Bar Chart Of occupation & Salary',height=300, width=500, color='occupation', color_discrete_sequence=px.colors.qualitative.Plotly)

    fig4 = px.bar(df, x='marital_status', y='salary', title='Bar Chart Of marital_status & Salary',height=300, width=500, color='marital_status', color_discrete_sequence=px.colors.qualitative.Plotly)

    # Display the chart using st.plotly_chart
    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.plotly_chart(fig4, use_container_width=True)

    # # fig5 = px.bar(df, x='occupation', y='salary', title='Sample Bar Chart',height=300, width=400)

    # # fig6 = px.bar(df, x='sex', y='salary', title='Sample Bar Chart',height=300, width=400)

    # # # Display the chart using st.plotly_chart
    # # col5, col6 = st.columns(2)

    # # with col5:
    # #     st.plotly_chart(fig5, use_container_width=True)

    # # with col6:
    # #     st.plotly_chart(fig6, use_container_width=True)

    # fig3 = px.line(df, x='salary', y=['capital_gain', 'capital_loss'], title='Sample Bar Chart')

    # # Display the chart using st.plotly_chart
    # st.plotly_chart(fig3, use_container_width=True)






def author_page():
    st.title("About Me")
    st.write("I hold a PGDM in Finance Management and have 2 years of experience in banking operations. I possess strong knowledge in data analysis and machine learning skills, and I am currently working in the field of Data Science. I am seeking an environment where I can nurture and challenge my passion for Data Science and proficiency in data analysis and machine learning. Please feel free to check out my profile below, and if you have any doubts, we can connect on LinkedIn.")
    st.write(":blond-haired-man:Name:-Milind Shende")
    st.write(":calling: MObile No:-+91 9420699550")
    st.write(":e-mail: E-mail:-milind.shende24rediffmail.com")
    st.write(":link:Github:-https://github.com/Milind-Shende/census.git")
    st.write(":link:Linkedin:-https://www.linkedin.com/in/milind-shende/")
    

# Main prediction page
def prediction_page():
    # Title and input fields
    st.title('Credit Card Default Predication')
    st.subheader('Customer Information')
    age = st.number_input('AGE', min_value=17, max_value=90, value=25, step=1)
    workclass = st.selectbox('Workclass', ('Private', 'State-gov','Without-pay'))
    fnlwgt = st.number_input('Final Weight')
    education = st.selectbox('Education', ('HS-grad', 'JR-grad','Graduate','Masters','Doctorate'))
    marital_status = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
    occupation = st.selectbox('Occupation', ('Adm-clerical', 'Exec-managerial','Prof-specialty', 'Sales', 'Craft-repair',
       'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Protective-serv', 'Armed-Forces','Other-service'))
    sex = st.selectbox('Sex', ('Male', 'Female'))
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=99999, value=0, step=1)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=4356, value=0, step=1)
    hours_per_week = st.number_input('Hours Per Week', min_value=0, max_value=99, value=40, step=1) 
    country = st.selectbox('Country', ('United-States', 'Other-States'))
    

    
    
     
    # Prediction button
    if st.button('Predict'):
        try:
            # Preprocess the input features
            input_data = {
                'age':[age],
                'workclass':[workclass],
                'fnlwgt':[fnlwgt],
                'education':[education],
                'marital_status':[marital_status],
                'occupation':[occupation],
                'sex':[sex],
                'capital_gain':[capital_gain],
                'capital_loss':[capital_loss],
                'hours_per_week':[hours_per_week],
                'country':[country]
                    
            }
        except Exception as e:
            st.error(f"Error occurred: {e}")
        # Convert input data to a Pandas DataFrame
        input_df = pd.DataFrame(input_data)
        # Perform the transformation using the loaded transformer
        transformed_data = transfomer.transform(input_df)
        # Reshape the transformed data as a NumPy array
        input_arr = np.array(transformed_data)
        

        # Make the prediction using the loaded model
        prediction = model.predict(input_arr)
        st.subheader('Prediction')
        prediction_text = ">50K" if prediction[0] == 1 else "<=50K"
        st.write(f'The predicted Salary is: {prediction_text}')

def image():
    # Load and display the Nigerian flag image
    image_url = "https://github.com/Milind-Shende/census/blob/main/Income-PNG-Free-Download.png?raw=true"
    image_width = 200
    st.sidebar.image(image_url, width=image_width)




# Create a dictionary with page names and their corresponding functions
pages = {
    'About': about_page,
    'Visualization':visualization_page,
    'Prediction': prediction_page,
    'Author':author_page
}

# Streamlit application
def main():
    # Sidebar navigation
    st.sidebar.title('Navigation')
    selected_page = st.sidebar.radio('Go to', list(pages.keys()))
    image()

    # Display the selected page content
    pages[selected_page]()

# Run the Streamlit application
if __name__ == '__main__':
    main()