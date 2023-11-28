import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# There are a couple of things I would want to improve:
# 1. Don't retrain the model when we update the "penguin to predict"
# 2. Look at how does the file uploader and the submit button can interact
# 3. Make this thing more predictable by caching some application states.

st.title("Penguin Classifier")
st.write(
    "This app uses 6 inputs to predict the species of penguin using"
    "a model built on the Palmer Penguins dataset. Use the form below"
    " to get started!"
)

# penguin_file = st.file_uploader('Upload your own penguin data')
if penguin_file is None:
    penguin_df = pd.read_csv("penguins.csv")
    with open("random_forest_penguin.pickle", "rb") as rf_pickle:
        rfc = pickle.load(rf_pickle)

    with open("output_penguin.pickle", "rb") as map_pickle:
        unique_penguin_mapping = pickle.load(map_pickle)
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df.dropna(inplace=True)
    output = penguin_df["species"]
    features = penguin_df[
        [
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
    ]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values, y_train)
    y_pred = rfc.predict(x_test.values)
    score = round(accuracy_score(y_pred, y_test), 3)
    st.write(
        f"""We trained a Random Forest model on these
             data, it has a score of {score}! Use the
             inputs below to try out the model"""
    )

with st.form("user_inputs"):
    island = st.selectbox("Penguin Island", options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input(
        "Bill Length (mm) [30 - 60]", min_value=30, max_value=60
    )
    bill_depth = st.number_input(
        "Bill Depth (mm) [10 - 25]", min_value=10, max_value=25
    )
    flipper_length = st.number_input(
        "Flipper Length (mm) [150 - 250]", min_value=150, max_value=250
    )
    body_mass = st.number_input(
        "Body Mass (g) [2500 - 6500]", min_value=2500, max_value=6500
    )
    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1
sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male,
        ]
    ]
)

prediction_species = unique_penguin_mapping[new_prediction][0]
st.subheader("Predicting Your Penguin's Species:")
st.write(f"We predict your penguin is of the {prediction_species} species")
st.write(
    """We used a machine learning (Random Forest)
    model to predict the species, the features
    used in this predictions are ranked by
    relative importance below."""
)
st.image("feature_importance.png")
st.write(
    """Below are the histograms for each 
    continuous variable separated by penguin 
    species. The vertical line represents 
    your the inputted value."""
)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_length_mm"], hue=penguin_df["species"])
plt.axvline(bill_length)
plt.title("Bill Length by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["bill_depth_mm"], hue=penguin_df["species"])
plt.axvline(bill_depth)
plt.title("Bill Depth by Species")
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df["flipper_length_mm"], hue=penguin_df["species"])
plt.axvline(flipper_length)
plt.title("Flipper Length by Species")
st.pyplot(ax)
