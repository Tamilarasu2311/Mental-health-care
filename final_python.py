import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from PIL import Image  # Import Image from PIL

def main():
    dt = pd.read_csv('After label encoding.csv')

    # Load the image
    img = Image.open('mental-health-2019924_640.jpg')

    # Display the image
    st.image(img)

    Country = ["Australia", "Belgium", "Bosnia and Herzegovina", "Brazil", "Canada",
               "Colombia", "Costa Rica", "Croatia", "Czech Republic", "Denmark",
               "Finland", "France", "Georgia", "Germany", "Greece", "India", "Ireland",
               "Israel", "Italy", "Mexico", "Moldova", "Netherlands", "New Zealand",
               "Nigeria", "Philippines", "Poland", "Portugal", "Russia", "Singapore",
               "South Africa", "Sweden", "Switzerland", "Thailand", "United Kingdom",
               "United States"]

    Occupation = ["Business", "Corporate", "Housewife", "Others", "Student"]

    self_employed = ['No', 'Yes']
    family_history = ['No', 'Yes']
    treatment = ['No', 'Yes']
    Days_Indoors = ["1-14 days", "15-30 days", "31-60 days", "Go out Every day", "More than 2 months"]
    Growing_Stress = ['Maybe', 'No', 'Yes']
    Changes_Habits = ['Maybe', 'No', 'Yes']
    Mental_Health_History = ['Maybe', 'No', 'Yes']
    Mood_Swings = ['High', 'Low', 'Medium']
    Coping_Struggles = ['No', 'Yes']
    Work_Interest = ['Maybe', 'No', 'Yes']
    Social_Weakness = ['Maybe', 'No', 'Yes']
    mental_health_interview = ['Maybe', 'No', 'Yes']

    custom_css = """
    <style>
    /* Change the background color of the sidebar */
    [data-testid="stSidebar"][aria-expanded="true"] {
        background: linear-gradient(to right, #ff8a00, #da1b60); /* Gradient color */
    }

    {
        background-image: https://www.remove.bg/upload; /* Relative path to the image file */
        background-size: cover; /* Cover the entire background */
        background-position: center; /* Center the background */
        background-repeat: no-repeat; /* Do not repeat the background */

    /* Change the text size of the sidebar elements */
    .st-af {
        font-size: 20px !important; /* Larger font size */
        color:black
    }
 
    /* Change the text size of the selectbox label */
    p{
        font-size: 80px !important; /* Larger font size */
        color:white;
    }
    
    /* Change the color of the predict button */
    div.stButton > button {
        background-color: green;
        color: white;
        font-weight: bold;
    }
    /* Change the color of the predict button on hover */
    div.stButton > button:hover {
        background-color: #FF474C;
    }

    </style>
    """

    # Inject custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Create a container to position main content and sidebar
    main_container = st.container()

    with main_container:
        with st.sidebar:
            st.markdown("<h1 style='text-align: center; font-size: 40px; color:#FFFFFF;'>User Input</h1>",
                        unsafe_allow_html=True)
            countries_input = st.selectbox('Country', Country)
            occupation_input = st.selectbox('occupation', Occupation)
            self_employed_input = st.selectbox('self_employed', self_employed)
            family_history_input = st.selectbox('family_history', family_history)
            treatment_input = st.selectbox('treatment', treatment)
            Days_Indoors_input = st.selectbox('Days_Indoors', Days_Indoors)
            Growing_Stress_input = st.selectbox('Growing_Stress', Growing_Stress)
            Changes_Habits_input = st.selectbox('Changes_Habits', Changes_Habits)
            Mental_Health_History_input = st.selectbox('Mental_Health_History', Mental_Health_History)
            Mood_Swings_input = st.selectbox('Mood_Swings', Mood_Swings)
            Coping_Struggles_input = st.selectbox('Coping_Struggles', Coping_Struggles)
            Work_Interest_input = st.selectbox('Work_Interest', Work_Interest)
            Social_Weakness_input = st.selectbox('Social_Weakness', Social_Weakness)
            mental_health_interview_input = st.selectbox('mental_health_interview', mental_health_interview)

        features = {
            'Country': [Country.index(countries_input)],
            'Occupation': [Occupation.index(occupation_input)],
            'self_employed': [self_employed.index(self_employed_input)],
            'family_history': [family_history.index(family_history_input)],
            'treatment': [treatment.index(treatment_input)],
            'Days_Indoors': [Days_Indoors.index(Days_Indoors_input)],
            'Growing_Stress': [Growing_Stress.index(Growing_Stress_input)],
            'Changes_Habits': [Changes_Habits.index(Changes_Habits_input)],
            'Mental_Health_History': [Mental_Health_History.index(Mental_Health_History_input)],
            'Mood_Swings': [Mood_Swings.index(Mood_Swings_input)],
            'Coping_Struggles': [Coping_Struggles.index(Coping_Struggles_input)],
            'Work_Interest': [Work_Interest.index(Work_Interest_input)],
            'Social_Weakness': [Social_Weakness.index(Social_Weakness_input)],
            'mental_health_interview': [mental_health_interview.index(mental_health_interview_input)]
        }

        features_df = pd.DataFrame(features)

        predict_button = st.button('Predict')

        # Dictionary mapping numerical predictions to labels
        prediction_labels = {0: 'Maybe', 1: 'No', 2: 'Yes'}

        if predict_button:
            # Create a Logistic Regression Classifier
            dt_classifier = LogisticRegression()

            x = dt.drop(['care_options'], axis=1)
            y = dt['care_options']

            dt_classifier.fit(x, y)

            predicted_out_dt = dt_classifier.predict(features_df)

            # Convert numerical predictions to labels
            predicted_labels = [prediction_labels[pred] for pred in predicted_out_dt]

            out = pd.DataFrame(predicted_labels, index=[0])
            st.write('Care options:' + ' ' + out.values.flatten())


if __name__ == "__main__":
    main()
