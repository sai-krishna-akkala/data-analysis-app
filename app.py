import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
import io

# -----------------------------
# App Configuration & Styling
# -----------------------------
st.set_page_config(page_title="Analyze Data With Clicks", page_icon="🔍", layout="wide")

def set_background():
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://img.freepik.com/free-vector/gradient-background-wave-design-minimalist-style_483537-3592.jpg?semt=ais_hybrid&w=740");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def marquee():
    marquee_html = """
    <marquee behavior="scroll" direction="left" scrollamount="10" style="font-size:24px; color:blue;">
      📊 Welcome to the Streamlit web App! 
    </marquee>
    """
    st.markdown(marquee_html, unsafe_allow_html=True)

def sidebar_style():
    st.markdown(
        """
        <style>
        .my-plot-container {
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Initialize styles
set_background()
marquee()
sidebar_style()
st.title("🔍 Analyze Data With Clicks")
st.write("Upload CSV or connect to SQL Server to start analyzing your data!")

# -----------------------------
# Data Loader Class
# -----------------------------
class DataLoader:
    def __init__(self):
        self.df = None
        self.conn = None

    def load_csv(self):
        file_uploaded = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if file_uploaded is not None:
            st.sidebar.success('File uploaded successfully')
            self.df = pd.read_csv(file_uploaded)

    def load_sql_server(self):
        server = st.sidebar.text_input('Server name', value='localhost\\SQLEXPRESS')
        use_sql_auth = st.sidebar.checkbox("Use SQL Authentication (username/password)", value=False)

        if use_sql_auth:
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
        else:
            username = password = None

        connect_btn = st.sidebar.button("Connect to SQL Server")
        if connect_btn:
            try:
                if use_sql_auth:
                    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};UID={username};PWD={password};"
                else:
                    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};Trusted_Connection=yes;"
                self.conn = pyodbc.connect(conn_str)
                st.sidebar.success("Connected to SQL Server!")
            except Exception as e:
                st.sidebar.error(f"Connection failed: {e}")

            if self.conn:
                try:
                    dbs = pd.read_sql("SELECT name FROM sys.databases", self.conn)
                    database = st.sidebar.selectbox('Choose database', dbs['name'].tolist())
                    # TODO: Load selected database into df
                except Exception as e:
                    st.sidebar.error(f"Error fetching databases: {e}")

# -----------------------------
# EDA Class
# -----------------------------
class EDA:
    def __init__(self, df):
        self.df = df

    def show_head(self):
        st.subheader("First 5 Rows")
        st.write(self.df.head())

    def show_tail(self):
        st.subheader("Last 5 Rows")
        st.write(self.df.tail())

    def show_shape(self):
        st.subheader("Shape")
        st.write(self.df.shape)

    def show_describe(self):
        st.subheader("Statistical Summary")
        st.write(self.df.describe())

    def show_info(self):
        st.subheader("Info")
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        st.text(buffer.getvalue())

    def show_missing(self):
        st.subheader("Missing Values")
        missing_values = self.df.isnull().sum()
        st.write(missing_values[missing_values > 0])

# -----------------------------
# Data Cleaning Class
# -----------------------------
class DataCleaner:
    def __init__(self, df):
        self.df = df

    def fill_missing(self):
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            col = st.selectbox("Select column with missing values", missing_cols)
            method = st.radio("Choose fill method", ["Mean", "Median", "Mode", "Custom Value"])
            if method == "Mean":
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif method == "Median":
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif method == "Mode":
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
            elif method == "Custom Value":
                custom_val = st.text_input("Enter custom value")
                if custom_val:
                    self.df[col].fillna(custom_val, inplace=True)
            st.success(f"Missing values in `{col}` filled using {method}")

    def convert_dtype(self):
        col = st.selectbox("Select column to convert", self.df.columns)
        dtype_choice = st.selectbox("Convert to type", ["int", "float", "str", "datetime", "category"])
        if st.button("Convert Type"):
            try:
                if dtype_choice == "datetime":
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                else:
                    self.df[col] = self.df[col].astype(dtype_choice)
                st.success(f"Converted `{col}` to `{dtype_choice}`")
            except Exception as e:
                st.error(f"Conversion failed: {e}")

# -----------------------------
# Visualization Class
# -----------------------------
class Visualizer:
    def __init__(self, df):
        self.df = df

    def bar_chart(self):
        cat_col = st.selectbox("Categorical column", self.df.select_dtypes('object').columns)
        num_cols = st.multiselect("Numeric columns", self.df.select_dtypes('number').columns)
        if st.button("Generate Bar Chart") and cat_col and num_cols:
            fig, ax = plt.subplots(figsize=(7,5))
            x = np.arange(len(self.df[cat_col]))
            bar_width = 0.8 / len(num_cols)
            for i, col in enumerate(num_cols):
                ax.bar(x + i*bar_width, self.df[col], width=bar_width, label=col)
            ax.set_xticks(x + bar_width*(len(num_cols)-1)/2)
            ax.set_xticklabels(self.df[cat_col], rotation=45)
            ax.legend()
            st.pyplot(fig)

    # Add other methods like pie_chart(), histogram(), scatter_plot(), heatmap() similarly

# -----------------------------
# Main App Logic
# -----------------------------
source = st.sidebar.selectbox("Choose Source", ["CSV File", "SQL Server"])
loader = DataLoader()
if source == "CSV File":
    loader.load_csv()
elif source == "SQL Server":
    loader.load_sql_server()

if loader.df is not None:
    eda = EDA(loader.df)
    st.subheader("🔍 EDA")
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1,1,1.5,1])
    if c1.button("Head"): eda.show_head()
    if c2.button("Tail"): eda.show_tail()
    if c3.button("Shape"): eda.show_shape()
    if c4.button("Describe"): eda.show_describe()
    if c5.button("Info"): eda.show_info()
    if c6.button("Missing Values"): eda.show_missing()
    if c7.button("View Data"): st.dataframe(loader.df)

    cleaner = DataCleaner(loader.df)
    st.subheader("🧹 Data Cleaning")
    cleaner.fill_missing()
    cleaner.convert_dtype()

    st.subheader("📊 Visualization")
    viz = Visualizer(loader.df)
    viz.bar_chart()
