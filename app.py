import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
import io
import seaborn as sns

# ---------------------- App Config ----------------------
st.set_page_config(page_title="Analyze Data With Clicks", page_icon="🔍", layout="wide")

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

st.markdown(
    """
    <style>
    .my-plot-container {
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("""
<marquee behavior="scroll" direction="left" scrollamount="10" style="font-size:24px; color:blue;">
  📊 Welcome to the Streamlit web App! 
</marquee>
""", unsafe_allow_html=True)

st.title("🔍 Analyze Data With Clicks")
st.write("This app allows you to perform EDA, data cleaning, and visualization on your data without writing any code.")

# ---------------------- Classes ----------------------
class DataLoader:
    def __init__(self):
        self.df = None

    def load_csv(self):
        file_uploaded = st.sidebar.file_uploader("Choose a CSV file", type="csv")
        if file_uploaded is not None:
            st.sidebar.success("File uploaded successfully")
            self.df = pd.read_csv(file_uploaded)

    def load_sql(self):
        server = st.sidebar.text_input('Server name', value='localhost\\SQLEXPRESS')
        use_sql_auth = st.sidebar.checkbox("Use SQL Authentication (username/password)", value=False)
        username = password = None
        if use_sql_auth:
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
        conn = None
        connect_btn = st.sidebar.button("Connect to SQL Server")
        if connect_btn:
            try:
                if use_sql_auth:
                    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};UID={username};PWD={password};"
                else:
                    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};Trusted_Connection=yes;"
                conn = pyodbc.connect(conn_str)
                st.sidebar.success("Connected to SQL Server!")
                dbs = pd.read_sql("SELECT name FROM sys.databases", conn)
                database = st.sidebar.selectbox('Choose database', dbs['name'].tolist())
                # Optional: query and load data here
            except Exception as e:
                st.sidebar.error(f"Connection failed: {e}")

class EDA:
    def __init__(self, df):
        self.df = df

    def display_options(self):
        st.subheader("🔍EDA")
        st.write("Perform basic EDA on your uploaded CSV file")
        c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1,1,1.5,1])
        self.buttons = {
            "head": c1.button("Head"),
            "tail": c2.button("Tail"),
            "shape": c3.button("Shape"),
            "describe": c4.button("Describe"),
            "info": c5.button("Info"),
            "missing": c6.button("Missing Values"),
            "view": c7.button("View Data")
        }
        self.show_output()

    def show_output(self):
        try:
            if self.buttons["head"]:
                st.subheader("First 5 Rows:")
                st.write(self.df.head())
            if self.buttons["tail"]:
                st.subheader("Last 5 Rows:")
                st.write(self.df.tail())
            if self.buttons["shape"]:
                st.subheader("Shape of the DataFrame:")
                st.write(self.df.shape)
            if self.buttons["describe"]:
                st.subheader("Statistical Summary:")
                st.write(self.df.describe())
            if self.buttons["info"]:
                st.subheader("Info:")
                buffer = io.StringIO()
                self.df.info(buf=buffer)
                st.text(buffer.getvalue())
            if self.buttons["missing"]:
                st.subheader("Missing Values:")
                missing_values = self.df.isnull().sum()
                st.write(missing_values[missing_values > 0])
            if self.buttons["view"]:
                st.subheader("Data")
                st.dataframe(self.df)
        except Exception:
            st.error("Please choose Data....")

class DataCleaner:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        b1, b2 = st.columns([1,1])
        with b1: st.subheader("🔽Filter Data")
        with b2: st.subheader("🧹Data Cleaning")
        f1, f2 = st.columns([1,1])
        filter_ = f1.button("Filter")
        clean_ = f2.button("Clean")

        # Filter toggle
        if "_show_filter_" not in st.session_state:
            st.session_state._show_filter_ = False
        if filter_:
            st.session_state._show_filter_ = not st.session_state._show_filter_
        if st.session_state._show_filter_:
            self.filter_data()

        if clean_:
            self.clean_missing_data()
            self.data_type_conversion()
            self.filter_rows()
            self.drop_rename_index()

    def filter_data(self):
        filter_type = st.selectbox("Choose Filter Type", ["Basic", "Custom"], index=0)
        if filter_type == "Basic":
            col = st.selectbox("Select column", self.df.columns)
            val = st.selectbox("Select value", self.df[col].unique())
            st.dataframe(self.df[self.df[col]==val])
        elif filter_type == "Custom":
            selected_cols = st.multiselect("Select columns to display", self.df.columns, default=self.df.columns)
            filters = {}
            for col in selected_cols:
                if self.df[col].dtype=='object':
                    filters[col] = st.multiselect(f"Filter {col}", self.df[col].unique(), default=self.df[col].unique())
                else:
                    min_val, max_val = self.df[col].min(), self.df[col].max()
                    filters[col] = st.slider(f"Range for {col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
            filtered_df = self.df.copy()
            for col, val in filters.items():
                if isinstance(val, list):
                    filtered_df = filtered_df[filtered_df[col].isin(val)]
                elif isinstance(val, tuple):
                    filtered_df = filtered_df[(filtered_df[col]>=val[0]) & (filtered_df[col]<=val[1])]
            st.dataframe(filtered_df[selected_cols])

    def clean_missing_data(self):
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            col_to_fill = st.selectbox("Select column with missing values", missing_cols)
            method = st.radio("Choose fill method", ["Mean", "Median", "Mode", "Custom Value"])
            if method=="Mean": self.df[col_to_fill].fillna(self.df[col_to_fill].mean(), inplace=True)
            elif method=="Median": self.df[col_to_fill].fillna(self.df[col_to_fill].median(), inplace=True)
            elif method=="Mode": self.df[col_to_fill].fillna(self.df[col_to_fill].mode()[0], inplace=True)
            elif method=="Custom Value":
                custom_val = st.text_input("Enter custom value")
                if custom_val: self.df[col_to_fill].fillna(custom_val, inplace=True)
            st.success(f"Filled missing values in {col_to_fill} using {method}")
            st.dataframe(self.df.head())

    def data_type_conversion(self):
        st.markdown("---")
        st.markdown("## Data Type Conversion")
        col_dtype = st.selectbox("Select column to convert", self.df.columns)
        dtype_choice = st.selectbox("Convert to type", ["int", "float", "str", "datetime", "category"])
        if st.button("Convert Type"):
            try:
                if dtype_choice=="datetime": self.df[col_dtype]=pd.to_datetime(self.df[col_dtype], errors='coerce')
                else: self.df[col_dtype]=self.df[col_dtype].astype(dtype_choice)
                st.success(f"Converted {col_dtype} to {dtype_choice}")
            except Exception as e:
                st.error(f"Conversion failed: {e}")

    def filter_rows(self):
        st.markdown("---")
        st.markdown("## Filter Rows")
        filter_col = st.selectbox("Select column to filter", self.df.columns)
        if self.df[filter_col].dtype=='object':
            val = st.selectbox("Select value", self.df[filter_col].unique())
            st.dataframe(self.df[self.df[filter_col]==val])
        else:
            min_val, max_val = float(self.df[filter_col].min()), float(self.df[filter_col].max())
            selected_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
            st.dataframe(self.df[(self.df[filter_col]>=selected_range[0]) & (self.df[filter_col]<=selected_range[1])])

    def drop_rename_index(self):
        st.markdown("---")
        drop_cols = st.multiselect("Select columns to drop", self.df.columns)
        drop_rows = st.multiselect("Select rows to drop", self.df.index.tolist())
        if st.button("Drop Selected"):
            self.df.drop(columns=drop_cols, inplace=True)
            self.df.drop(index=drop_rows, inplace=True)
            st.success("Dropped selected columns/rows")
        st.markdown("---")
        rename_col = st.selectbox("Select column to rename", self.df.columns)
        new_name = st.text_input("New column name")
        if st.button("Rename Column"):
            self.df.rename(columns={rename_col:new_name}, inplace=True)
            st.success(f"Renamed {rename_col} to {new_name}")
        st.markdown("---")
        idx_action = st.radio("Index Action", ["Reset Index", "Set Index"])
        if idx_action=="Reset Index":
            self.df.reset_index(drop=True, inplace=True)
            st.success("Index reset")
        else:
            idx_col = st.selectbox("Select column to set as index", self.df.columns)
            self.df.set_index(idx_col, inplace=True)
            st.success(f"Set {idx_col} as index")
        st.subheader("🧾 Cleaned Data")
        st.dataframe(self.df)

class Visualization:
    def __init__(self, df):
        self.df = df
        if "hidden_charts" not in st.session_state:
            st.session_state.hidden_charts = {v: False for v in ["Bar Chart","Line Chart","Pie Chart","Histogram","Box Plot","Scatter Plot","Heatmap"]}

    def visualize(self):
        st.subheader("📊 Visualization")
        viz_type = st.selectbox("Select Visualization", ["None","Bar Chart","Pie Chart","Histogram","Box Plot","Scatter Plot","Heatmap"])
        if viz_type=="None": return

        v1,v2 = st.columns([1,1])
        with v1: self.generate_chart(viz_type)
        with v2: self.show_description(viz_type)

    def generate_chart(self, viz_type):
        st.markdown(f"### {viz_type}")
        if viz_type=="Bar Chart":
            categorical_col = st.selectbox("Categorical Column", self.df.select_dtypes(include=['object','category']).columns)
            numeric_cols = st.multiselect("Numeric Columns", self.df.select_dtypes(include='number').columns)
            if categorical_col and numeric_cols and st.button("Generate Chart"):
                fig, ax = plt.subplots(figsize=(7,5))
                x = np.arange(len(self.df[categorical_col]))
                bar_width = 0.8/len(numeric_cols)
                for i,col in enumerate(numeric_cols):
                    ax.bar(x+i*bar_width, self.df[col], width=bar_width, label=col)
                ax.set_xticks(x+bar_width*(len(numeric_cols)-1)/2)
                ax.set_xticklabels(self.df[categorical_col], rotation=45, ha='right')
                ax.set_title(f"{categorical_col} vs {', '.join(numeric_cols)}")
                ax.set_xlabel(categorical_col)
                ax.set_ylabel("Values")
                ax.legend()
                st.pyplot(fig)

        elif viz_type=="Pie Chart":
            cat_col = st.selectbox("Categorical Column", self.df.columns)
            num_col = st.selectbox("Numerical Column", self.df.columns)
            if cat_col and num_col and st.button("Generate Pie Chart"):
                fig, ax = plt.subplots(figsize=(7,7))
                ax.pie(self.df[num_col], labels=self.df[cat_col], autopct="%1.1f%%")
                ax.axis("equal")
                st.pyplot(fig)
        # Similarly implement Histogram, Box Plot, Scatter Plot, Heatmap here (omitted for brevity)
        
    def show_description(self, viz_type):
        descriptions = {
            "Bar Chart": "A bar chart compares categories using bars.",
            "Pie Chart": "A pie chart shows relative proportions of categories.",
            "Histogram": "A histogram shows frequency distribution of numeric data.",
            "Box Plot": "A box plot shows median, quartiles, and outliers.",
            "Scatter Plot": "A scatter plot shows relationships between two numeric variables.",
            "Heatmap": "A heatmap shows correlation patterns using color intensity."
        }
        st.header("Description")
        st.write(descriptions.get(viz_type,""))

# ---------------------- Main Logic ----------------------
source = st.sidebar.selectbox("Choose Source", ["CSV File","SQL Server"])
loader = DataLoader()
if source=="CSV File": loader.load_csv()
else: loader.load_sql()

if loader.df is not None:
    eda = EDA(loader.df)
    eda.display_options()

    cleaner = DataCleaner(loader.df)
    cleaner.clean_data()

    viz = Visualization(loader.df)
    viz.visualize()
