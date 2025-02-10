import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config (MUST be the first Streamlit command)
st.set_page_config(layout="wide")  

# Load dataset with caching
@st.cache_data
def load_data():
    return sns.load_dataset('titanic')

df = load_data()

# Create two columns
col1, col2 = st.columns([0.2, 0.8])  # Adjust column widths

# Add image in the first column
with col1:
    st.image("titanic.png", width=200)  # Ensure "titanic.png" is in the same directory

# Add title in the second column
with col2:
    st.title("Titanic Data Analysis")

# Show Dataset
st.subheader("Dataset Overview")
st.write(df.head())

# Summary statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Data Visualizations
st.subheader("Graphs and Analysis")

# 1️⃣ Class Distribution
st.write("### Class Distribution")
fig, ax = plt.subplots(figsize=(4, 3))
sns.countplot(data=df, x="class", palette="pastel", ax=ax)
st.pyplot(fig)

# 2️⃣ Survival Rate by Gender
st.write("### Survival Rate by Gender")
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x="sex", y="survived", data=df, errorbar=None, palette="coolwarm", ax=ax)
st.pyplot(fig)

# 3️⃣ Passenger Count by Embarkation Port
st.write("### Passenger Count by Embarkation Port")
fig, ax = plt.subplots(figsize=(4, 3))
sns.countplot(data=df, x="embark_town", palette="muted", ax=ax)
st.pyplot(fig)

# 4️⃣ Survival Rate by Passenger Class
st.write("### Survival Rate by Passenger Class")
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x="class", y="survived", data=df, errorbar=None, palette="Blues", ax=ax)
st.pyplot(fig)

# 5️⃣ Age Distribution
st.write("### Age Distribution")
fig, ax = plt.subplots(figsize=(4, 3))
sns.histplot(df['age'].dropna(), kde=True, bins=30, ax=ax, color="blue")
st.pyplot(fig)

# 6️⃣ Survival Rate by Age Group
st.write("### Survival Rate by Age Group")
df["age_group"] = pd.cut(df["age"], bins=[0, 12, 18, 30, 50, 80], labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
fig, ax = plt.subplots(figsize=(4, 3))
sns.barplot(x="age_group", y="survived", data=df, errorbar=None, palette="viridis", ax=ax)
st.pyplot(fig)

# 7️⃣ Fare Distribution
st.write("### Fare Distribution")
fig, ax = plt.subplots(figsize=(4, 3))
sns.histplot(df['fare'], kde=True, bins=30, ax=ax, color="green")
st.pyplot(fig)

# 8️⃣ Boxplot for Fare by Class
st.write("### Fare Distribution by Passenger Class")
fig, ax = plt.subplots(figsize=(4, 3))
sns.boxplot(data=df, x="class", y="fare", palette="coolwarm", ax=ax)
st.pyplot(fig)

# 9️⃣ Boxplot for Age by Class
st.write("### Age Distribution by Passenger Class")
fig, ax = plt.subplots(figsize=(4, 3))
sns.boxplot(data=df, x="class", y="age", palette="muted", ax=ax)
st.pyplot(fig)

# 🔟 Correlation Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("🔍 Explore Data with Filters")

# Passenger Class filter
selected_class = st.selectbox("Select Passenger Class:", df["class"].unique())

# Gender filter
selected_gender = st.radio("Select Gender:", df["sex"].unique())

# Age Range filter using slider
age_range = st.slider("Select Age Range:", 
                     min_value=int(df["age"].min()),
                     max_value=int(df["age"].max()),
                     value=(0, int(df["age"].max())))

# Survival Status filter
selected_survived = st.multiselect("Survival Status:",
                                 options=["Survived", "Did not survive"],
                                 default=["Survived", "Did not survive"])

# Convert survival selection to boolean for filtering
survival_map = {"Survived": True, "Did not survive": False}
survival_values = [survival_map[status] for status in selected_survived]

# Apply all filters
filtered_df = df[
    (df["class"] == selected_class) & 
    (df["sex"] == selected_gender) &
    (df["age"].between(age_range[0], age_range[1])) &
    (df["survived"].isin(survival_values))
]

# Display filtered dataframe
st.write(filtered_df)

# Show number of results
st.text(f"Showing {len(filtered_df)} passengers matching the selected criteria")