import streamlit as st
import joblib
import pandas as pd

model = joblib.load('score_predictor.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

cat_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus',
            'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
            'famsup', 'paid', 'activities', 'nursery',
            'higher', 'internet', 'romantic']

num_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime',
            'failures', 'famrel', 'freetime', 'goout',
            'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']

st.title("Student Score Predictor (Portuguese subject)")
st.subheader("Categorical Features")

input_data = {}
input_data['school'] = st.selectbox("School", ['GP', 'MS'])
input_data['sex'] = st.selectbox("Sex", ['F', 'M'])
input_data['address'] = st.selectbox("Address", ['U', 'R'])
input_data['famsize'] = st.selectbox("Family Size", ['LE3', 'GT3'])
input_data['Pstatus'] = st.selectbox("Parent Status", ['T', 'A'])
input_data['Mjob'] = st.selectbox("Mother Job", ['teacher','health','services','at_home','other'])
input_data['Fjob'] = st.selectbox("Father Job", ['teacher','health','services','at_home','other'])
input_data['reason'] = st.selectbox("Reason", ['home','reputation','course','other'])
input_data['guardian'] = st.selectbox("Guardian", ['mother','father','other'])

binary_cols = ['schoolsup','famsup','paid','activities',
               'nursery','higher','internet','romantic']
for col in binary_cols:
    input_data[col] = st.selectbox(col, ['yes','no'])

st.subheader("Numerical Features")
for col in num_cols:
    input_data[col] = st.number_input(col, step=1)
if st.button("Predict"):

    input_df = pd.DataFrame([input_data])

    # Encode categorical
    encoded = ohe.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))

    # Scale numerical
    scaled = sc.transform(input_df[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=num_cols)

    # Combine in same order as training
    final_input = np.hstack((encoded, scaled))

    prediction = model.predict(final_input)[0]

    st.subheader("Predicted G3:")
    st.write(prediction)
