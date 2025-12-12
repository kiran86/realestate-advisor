# streamlit_app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer
import io

sns.set(style="whitegrid")

st.set_page_config(layout="wide", page_title="Real Estate EDA", initial_sidebar_state="expanded")

@st.cache_data
def load_data(path=None, uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if path is not None:
        # return pd.read_csv(path)
        return pd.read_csv("../data/india_housing_prices.csv")
    raise ValueError("Provide a path or upload a file.")

def prepare_amenities(df):
    # Normalize the Amenities column: remove brackets/quotes if present and split
    if "Amenities" not in df.columns:
        return df, []
    # Ensure str type
    df["Amenities"] = df["Amenities"].astype(str).replace("nan", "")
    # remove surrounding [] or quotes if any
    df["Amenities"] = df["Amenities"].str.replace(r"[\[\]'\"]", "", regex=True)
    df["Amenities_list"] = df["Amenities"].apply(lambda x: [a.strip() for a in x.split(",") if a.strip() != ""])
    mlb = MultiLabelBinarizer()
    if len(df["Amenities_list"].map(len).unique()) == 1 and df["Amenities_list"].map(len).unique()[0] == 0:
        # no amenities found
        amenity_df = pd.DataFrame(index=df.index)
        amenity_cols = []
    else:
        amenity_df = pd.DataFrame(mlb.fit_transform(df["Amenities_list"]), columns=mlb.classes_, index=df.index)
        amenity_cols = mlb.classes_.tolist()
    df = pd.concat([df, amenity_df], axis=1)
    if "Price_per_SqFt" not in df.columns and "Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns:
        df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]
    return df, amenity_cols

def plot_age_vs_price_scatter(df):
    plt.figure(figsize=(10,6))
    # alpha and small point size for dense plots
    sns.scatterplot(data=df, x="Age_of_Property", y="Price_per_SqFt", alpha=0.6, s=40, hue=df.get("BHK"))
    sns.regplot(data=df, x="Age_of_Property", y="Price_per_SqFt", scatter=False, color="red", lowess=True)
    plt.title("Age of Property vs Price per SqFt")
    plt.xlabel("Age of Property (years)")
    plt.ylabel("Price per SqFt")
    plt.legend(title="BHK", loc="upper right")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_age_group_boxplot(df):
    bins = [0,5,10,20,50,100]
    labels = ["0-5","6-10","11-20","21-50","50+"]
    df["Age_Group"] = pd.cut(df["Age_of_Property"].fillna(0), bins=bins, labels=labels, include_lowest=True)
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="Age_Group", y="Price_per_SqFt", palette="Set2")
    plt.title("Price per SqFt Across Age Groups")
    plt.xlabel("Age Group")
    plt.ylabel("Price per SqFt")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_amenity_bar(df, amenity_cols):
    if not amenity_cols:
        st.info("No amenity columns detected.")
        return
    amenity_price = {a: df[df[a] == 1]["Price_per_SqFt"].mean() for a in amenity_cols}
    amenity_price = pd.Series(amenity_price).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=amenity_price.index, y=amenity_price.values)
    plt.xticks(rotation=45)
    plt.ylabel("Average Price per SqFt")
    plt.title("Average Price per SqFt by Amenity")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_amenity_boxplots(df, amenity_cols):
    if not amenity_cols:
        st.info("No amenity columns detected.")
        return
    for a in amenity_cols:
        plt.figure(figsize=(8,4))
        sns.boxplot(x=df[a], y=df["Price_per_SqFt"])
        plt.title(f"Price per SqFt with vs without '{a}'")
        plt.xlabel(f"{a} present? 0=No 1=Yes")
        plt.ylabel("Price per SqFt")
        st.pyplot(plt.gcf())
        plt.clf()

def plot_amenity_count_vs_price(df, amenity_cols):
    if not amenity_cols:
        st.info("No amenity columns detected.")
        return
    df["Amenity_Count"] = df[amenity_cols].sum(axis=1)
    plt.figure(figsize=(10,6))
    sns.boxplot(x="Amenity_Count", y="Price_per_SqFt", data=df)
    plt.title("Amenity Count vs Price per SqFt")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_correlation_heatmap(df, amenity_cols):
    cols = list(amenity_cols) + ["Price_per_SqFt"]
    sub = df[cols].copy()
    corr = sub.corr()
    plt.figure(figsize=(8, max(4, len(cols)*0.2)))
    sns.heatmap(corr[["Price_per_SqFt"]].sort_values(by="Price_per_SqFt", ascending=False),
                annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation of Amenities with Price_per_SqFt")
    st.pyplot(plt.gcf())
    plt.clf()

def correlation_overview(df):
    st.subheader("Numeric Correlation Overview")
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.shape[1] > 1:
        plt.figure(figsize=(12,8))
        sns.heatmap(numeric.corr(), cmap="coolwarm")
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.write("Not enough numeric columns for correlation heatmap.")

# --- UI ---
st.title("Real Estate EDA")
st.write("Interactive EDA. Upload a CSV or use the default dataset path.")

# Sidebar controls
st.sidebar.header("Load data")
use_uploaded = st.sidebar.checkbox("Upload CSV instead of default file", value=False)
uploaded_file = None
if use_uploaded:
    uploaded_file = st.sidebar.file_uploader("Upload your india_housing_prices.csv", type=["csv"])
data_path = st.sidebar.text_input("Or enter dataset path", value="../data/india_housing_prices.csv")

if use_uploaded:
    if uploaded_file is None:
        st.warning("Upload a CSV file to proceed")
        st.stop()
    df = load_data(uploaded_file=uploaded_file)
else:
    try:
        # df = load_data(path=data_path)
        df = load_data("../data/india_housing_prices.csv")
    except Exception as e:
        st.error(f"Failed to load dataset from {data_path}: {e}")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.header("Quick actions")
if st.sidebar.button("Show raw data (head)"):
    st.dataframe(df.head())

# Basic cleaning & derived columns
if "Age_of_Property" not in df.columns and "Year_Built" in df.columns:
    df["Age_of_Property"] = 2025 - df["Year_Built"]
if "Price_per_SqFt" not in df.columns and "Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns:
    df["Price_per_SqFt"] = df["Price_in_Lakhs"] * 100000 / df["Size_in_SqFt"]

df, amenity_cols = prepare_amenities(df)

# Main layout: tabs
tabs = st.tabs(["Overview", "Price & Size", "Age vs Price", "Amenities", "Correlations", "Data"])

with tabs[0]:
    st.header("Dataset Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.write("Column types:")
    col_info = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)}).reset_index(drop=True)
    st.dataframe(col_info)
    missing = df.isnull().sum().sort_values(ascending=False)
    st.write("Missing values (top 20):")
    st.dataframe(missing.head(20))

with tabs[1]:
    st.header("Price & Size")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price distribution")
        fig = plt.figure(figsize=(8,4))
        sns.histplot(df["Price_in_Lakhs"].dropna(), bins=50, kde=True)
        st.pyplot(fig)
        plt.clf()
    with col2:
        st.subheader("Size distribution")
        fig = plt.figure(figsize=(8,4))
        sns.histplot(df["Size_in_SqFt"].dropna(), bins=50, kde=True)
        st.pyplot(fig)
        plt.clf()

with tabs[2]:
    st.header("Age vs Price")
    st.write("Choose visualization:")
    viz = st.selectbox("Visualization", ["Scatter + LOESS", "Boxplot (Age Groups)", "Line trend (smoothed)"])
    if viz == "Scatter + LOESS":
        st.subheader("Scatter plot with LOWESS trend")
        plot_age_vs_price_scatter(df)
    elif viz == "Boxplot (Age Groups)":
        st.subheader("Price per SqFt across Age groups")
        plot_age_group_boxplot(df)
    else:
        st.subheader("Smoothed trend line")
        plt.figure(figsize=(10,6))
        tmp = df.dropna(subset=["Age_of_Property", "Price_per_SqFt"]).sort_values("Age_of_Property")
        sns.lineplot(data=tmp, x="Age_of_Property", y="Price_per_SqFt")
        plt.title("Trend: Age vs Price per SqFt")
        st.pyplot(plt.gcf())
        plt.clf()

with tabs[3]:
    st.header("Amenities Analysis")
    st.write("Amenities detected:", amenity_cols if amenity_cols else "None")
    st.subheader("Average price per amenity")
    plot_amenity_bar(df, amenity_cols)
    st.subheader("Price distribution with/without each amenity")
    if st.checkbox("Show boxplots for each amenity (may be many plots)"):
        plot_amenity_boxplots(df, amenity_cols)
    st.subheader("Amenity count vs Price")
    plot_amenity_count_vs_price(df, amenity_cols)

with tabs[4]:
    st.header("Correlations")
    correlation_overview(df)
    st.subheader("Amenity vs Price correlation (if amenities present)")
    plot_correlation_heatmap(df, amenity_cols)

with tabs[5]:
    st.header("Data")
    st.write("Preview and download cleaned data")
    st.dataframe(df.head(200))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download cleaned CSV", data=buf.getvalue(), file_name="cleaned_india_housing_prices.csv", mime="text/csv")

st.sidebar.markdown("---")
st.sidebar.write("App converts your EDA notebook into interactive visualizations.")
st.sidebar.write("Reference: project description used for guidance.")
