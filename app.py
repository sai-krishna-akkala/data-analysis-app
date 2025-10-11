import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyodbc
import io

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
# Paste this once at the top of your app (or before any plotting)
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
marquee_html = """
<marquee behavior="scroll" direction="left" scrollamount="10" style="font-size:24px; color:blue;">
  📊 Welcome to the Streamlit web App! 
</marquee>
"""

st.markdown(marquee_html, unsafe_allow_html=True)
st.title("🔍 Analyze Data With Clicks")
st.write("This app allows you to perform  EDA, data cleaning, and visualization on your data without writing any code. Just upload your CSV file or connect to a SQL Server database and start exploring!")
st.sidebar.subheader("Choose Source")
source=st.sidebar.selectbox("Choose Source",["SQL Server", "CSV File"])
# --- SQL Server connection ---
if source == "CSV File":
    
    file_uploaded = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    # Only proceed if a file is uploaded
    if file_uploaded is not None:
        st.sidebar.success('File uploaded successfully')

        # Read CSV into DataFrame
        df = pd.read_csv(file_uploaded)
if source == "SQL Server":
    server = st.sidebar.text_input('Server name', value='localhost\\SQLEXPRESS')
    use_sql_auth = st.sidebar.checkbox("Use SQL Authentication (username/password)", value=False)

    if use_sql_auth:
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
    else:
        username = password = None

    conn = None
    connect_btn = st.sidebar.button("Connect to SQL Server")

    if connect_btn:
        if use_sql_auth and (not username or not password):
            st.sidebar.warning("Please enter username and password.")
        else:
            try:
                if use_sql_auth:
                    conn_str = (
                        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                        f"SERVER={server};"
                        f"UID={username};"
                        f"PWD={password};"
                    )
                else:
                    conn_str = (
                        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                        f"SERVER={server};"
                        f"Trusted_Connection=yes;"
                    )
                conn = pyodbc.connect(conn_str)
                st.sidebar.success("Connected to SQL Server!")
            except Exception as e:
                st.sidebar.error(f"Connection failed: {e}")

    if conn:
        try:
            dbs = pd.read_sql("SELECT name FROM sys.databases", conn)
            database = st.sidebar.selectbox('Choose database', dbs['name'].tolist())
            # Further steps...
        except Exception as e:
            st.sidebar.error(f"Error fetching databases: {e}")
st.subheader("🔍EDA")
st.write("Perform basic EDA on your uploaded CSV file")
c1, c2, c3, c4, c5, c6 ,c7= st.columns([1,1,1,1,1,1.5,1])

    # Track which button is clicked
head_btn = c1.button('Head')
tail_btn = c2.button('Tail')
shape_btn = c3.button('Shape')
describe_btn = c4.button('Describe')
info_btn = c5.button('Info')
missing_btn = c6.button('Missing Values')
view_btn = c7.button("View Data")

# Show output in full width below
try:
    
    if head_btn:
        st.subheader("First 5 Rows:")
        st.write(df.head())
        if st.button("clear"):
            st.session_state.show_data = False

    if tail_btn:
        st.subheader("Last 5 Rows:")
        st.write(df.tail())
        if st.button("clear"):
            st.session_state.show_data = False

    if shape_btn:
        st.subheader("Shape of the DataFrame:")
        st.write(df.shape)
        if st.button("clear"):
            st.session_state.show_data = False

    if describe_btn:
        st.subheader("Statistical Summary:")
        st.write(df.describe())
        if st.button("clear"):
            st.session_state.show_data = False

    if info_btn:
        st.subheader("Info:")
        # Capture the output of df.info()
        buffer = st.empty()
        import io
        info_buf = io.StringIO()
        df.info(buf=info_buf)
        st.text(info_buf.getvalue())
        if st.button("clear"):
            st.session_state.show_data = False
    if missing_btn:
        st.subheader("Missing Values:")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0])
    if view_btn:
        st.subheader("Data")
        st.dataframe(df)
        if st.button("clear"):
            st.session_state.show_data = False
except Exception:
    st.error("Please choose Data....")
b1,b2=st.columns([1,1])
with b1:
    st.subheader("🔽Filter Data")
with b2:
    st.subheader("🧹Data Cleaning")
f1,f2=st.columns([1,1])
filter_ = f1.button("Filter")
clean_ = f2.button("Clean")
# 🔄 Initialize filter toggle state
if "_show_filter_" not in st.session_state:
    st.session_state._show_filter_ = False

# 🔘 Toggle button
if filter_:
    st.session_state._show_filter_ = not st.session_state._show_filter_

# ✅ Show Filters
if st.session_state._show_filter_:
    filter_type = st.selectbox("Choose Filter Type", ["Basic", "Custom"], index=0)

    if filter_type == "Basic":
        st.markdown("### Basic Filter")
        col = st.selectbox("Select column", df.columns)
        val = st.selectbox("Select value", df[col].unique())
        filtered_df = df[df[col] == val]
        st.dataframe(filtered_df)

    elif filter_type == "Custom":
        st.markdown("### Custom Filter")
        selected_cols = st.multiselect("Select columns to display", df.columns, default=df.columns)

        filters = {}
        for col in selected_cols:
            if df[col].dtype == 'object':
                options = df[col].unique().tolist()
                filters[col] = st.multiselect(f"Filter by {col}", options, default=options)
            elif np.issubdtype(df[col].dtype, np.number):
                min_val, max_val = df[col].min(), df[col].max()
                filters[col] = st.slider(f"Range for {col}", float(min_val), float(max_val), (float(min_val), float(max_val)))

        filtered_df = df.copy()
        for col, val in filters.items():
            if isinstance(val, list):
                filtered_df = filtered_df[filtered_df[col].isin(val)]
            elif isinstance(val, tuple):
                filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]

        st.dataframe(filtered_df[selected_cols])
    if st.button("hide"):
            st.session_state._show_filter_ = False


if clean_:
    
    #if st.button("clean"):
        st.session_state.show_cleaning = True
        
        if 'show_cleaning_miss' not in st.session_state:
            st.session_state.show_cleaning_miss = True
        if st.session_state.show_cleaning_miss:
        
            missing_cols = df.columns[df.isnull().any()].tolist()
            try:
                if missing_cols:
                    col_to_fill = st.selectbox("Select column with missing values", missing_cols)
                    method = st.radio("Choose fill method", ["Mean", "Median", "Mode", "Custom Value"])
                    if method == "Mean":
                        df[col_to_fill].fillna(df[col_to_fill].mean(), inplace=True)
                    elif method == "Median":
                        df[col_to_fill].fillna(df[col_to_fill].median(), inplace=True)
                    elif method == "Mode":
                        df[col_to_fill].fillna(df[col_to_fill].mode()[0], inplace=True)
                    elif method == "Custom Value":
                        custom_val = st.text_input("Enter custom value")
                        if custom_val:
                          df[col_to_fill].fillna(custom_val, inplace=True)
                    st.success(f"Missing values in `{col_to_fill}` filled using {method}")
                    st.write(df.head())

                else:
                  st.info("No missing values in your data.")
            except Exception as e:
                st.error(f"Error in missing data imputation: {e}")
            
            st.markdown("---")
            st.markdown("## 2. Data Type Conversion")
            col_dtype = st.selectbox("Select column to convert", df.columns)
            dtype_choice = st.selectbox("Convert to type", ["int", "float", "str", "datetime", "category"])
            if st.button("Convert Type"):
                try:
                    if dtype_choice == "datetime":
                        df[col_dtype] = pd.to_datetime(df[col_dtype], errors='coerce')
                    else:
                        df[col_dtype] = df[col_dtype].astype(dtype_choice)
                    st.success(f"Converted `{col_dtype}` to `{dtype_choice}`")
                except Exception as e:
                    st.error(f"Conversion failed: {e}")

            st.markdown("---")
            st.markdown("## 3. Filter Rows")
            filter_col = st.selectbox("Select column to filter", df.columns)
            if df[filter_col].dtype == 'object':
                filter_val = st.selectbox("Select value", df[filter_col].unique())
                filtered_df = df[df[filter_col] == filter_val]
            else:
                min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                selected_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
                filtered_df = df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])]
            st.dataframe(filtered_df)

            st.markdown("---")
            st.markdown("## 4. Drop Columns / Rows")
            drop_cols = st.multiselect("Select columns to drop", df.columns)
            drop_rows = st.multiselect("Select row indices to drop", df.index.tolist())
            if st.button("Drop Selected"):
                df.drop(columns=drop_cols, inplace=True)
                df.drop(index=drop_rows, inplace=True)
                st.success("Selected columns and rows dropped.")

            st.markdown("---")
            st.markdown("## 5. Rename Columns")
            rename_col = st.selectbox("Select column to rename", df.columns)
            new_name = st.text_input("Enter new column name")
            if st.button("Rename Column"):
                df.rename(columns={rename_col: new_name}, inplace=True)
                st.success(f"Renamed `{rename_col}` to `{new_name}`")

            st.markdown("---")
            st.markdown("## 6. Reset or Set Index")
            index_action = st.radio("Choose Index Action", ["Reset Index", "Set Index"])
            if index_action == "Reset Index":
                df.reset_index(drop=True, inplace=True)
                st.success("Index reset.")
            else:
                idx_col = st.selectbox("Select column to set as index", df.columns)
                df.set_index(idx_col, inplace=True)
                st.success(f"Set `{idx_col}` as index.")

            st.markdown("---")
            st.subheader("🧾 Cleaned Data")
            st.dataframe(df)
            if st.button("hide"):
                #st.session_state.show_cleaning = False
                st.session_state.show_cleaning_miss = False
if "hidden_charts" not in st.session_state:
    st.session_state.hidden_charts = {v: False for v in [
        "Bar Chart", "Line Chart", "Pie Chart", "Histogram",
        "Box Plot", "Scatter Plot", "Heatmap"
    ]}

st.subheader("📊 Visualization")
viz_type = st.selectbox(
    "Select Visualization",
    ["None", "Bar Chart", "Pie Chart", "Histogram", "Box Plot", "Scatter Plot", "Heatmap"],
    index=0
)
v1,v2=st.columns([1,1])
with v1:
    if viz_type != "None" and not st.session_state.hidden_charts[viz_type]:
        st.markdown(f"### {viz_type}")


        if viz_type == "Bar Chart":
            categorical_col = st.selectbox(
                "Select categorical column (X-axis)",
                df.select_dtypes(include=['object', 'category']).columns  # only categorical/text columns
            )

            numeric_cols = st.multiselect(
                "Select numeric columns for clustered bars",
                df.select_dtypes(include='number').columns  # only numeric columns
            )

            if categorical_col and numeric_cols:
                if st.button("Generate Chart"):
                    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

                    x = np.arange(len(df[categorical_col]))
                    bar_width = 0.8 / len(numeric_cols)

                    for i, col in enumerate(numeric_cols):
                        ax.bar(
                            x + i * bar_width,
                            df[col],
                            width=bar_width,
                            label=col
                        )

                    ax.set_xticks(x + bar_width * (len(numeric_cols) - 1) / 2)
                    ax.set_xticklabels(df[categorical_col], rotation=45, ha='right')

                    ax.set_title(f"Clustered Column Chart: {categorical_col} vs {', '.join(numeric_cols)}", fontsize=12)
                    ax.set_xlabel(categorical_col)
                    ax.set_ylabel("Values")
                    ax.legend()

                    for i, col in enumerate(numeric_cols):
                        for j, val in enumerate(df[col]):
                            ax.text(j + i * bar_width, val, f"{val}", ha='center', va='bottom', fontsize=8)

                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Select one categorical column and at least one numeric column.")

        elif viz_type == "Pie Chart":
            cat_col = st.selectbox("Select categorical column", df.columns)
            num_col = st.selectbox("Select numerical column for values", df.columns)

            if cat_col and num_col:
                if st.button("Generate Pie Chart"):
                    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
                    ax.pie(
                        df[num_col],
                        labels=df[cat_col].astype(str),
                        autopct='%1.1f%%'
                    )
                    ax.axis('equal')
                    st.pyplot(fig)

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)

                    st.download_button("📥 Download Chart as PNG", data=buf.getvalue(),
                                    file_name="pie_chart.png", mime="image/png")
            else:
                st.warning("Please select both categorical and numerical columns.")

        elif viz_type == "Histogram":
            selected_cols = st.multiselect("Select numeric columns", df.select_dtypes(include='number').columns)
            bins = st.slider("Number of bins", 5, 50, 10)
            if selected_cols:
                if st.button("Generate Histogram"):
                    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
                    for col in selected_cols:
                        ax.hist(df[col], bins=bins, alpha=0.5, label=col)
                    ax.set_title("Histogram")
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Frequency")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Select one or more numeric columns.")

        elif viz_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=['int', 'float']).columns
            selected_cols = st.multiselect("Select numeric columns", numeric_cols)

            if selected_cols:
                if st.button("Generate Box Plot"):
                    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
                    bp = ax.boxplot([df[col].dropna() for col in selected_cols],
                                    labels=selected_cols,
                                    patch_artist=True)

                    for i, col in enumerate(selected_cols, start=1):
                        data = df[col].dropna()
                        min_val = data.min()
                        q1 = data.quantile(0.25)
                        median = data.median()
                        q3 = data.quantile(0.75)
                        max_val = data.max()

                        ax.text(i, min_val, f"Min: {min_val:.2f}", ha='center', va='top', fontsize=8, color="blue")
                        ax.text(i, q1, f"Q1: {q1:.2f}", ha='center', va='bottom', fontsize=8, color="green")
                        ax.text(i, median, f"Median: {median:.2f}", ha='center', va='bottom', fontsize=8, color="red")
                        ax.text(i, q3, f"Q3: {q3:.2f}", ha='center', va='bottom', fontsize=8, color="green")
                        ax.text(i, max_val, f"Max: {max_val:.2f}", ha='center', va='bottom', fontsize=8, color="blue")

                    ax.set_title("Box Plot with Stats")
                    ax.set_ylabel("Values")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Select one or more numeric columns.")

        elif viz_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis", df.columns)
            y_col = st.selectbox("Select Y-axis", df.columns)
            hue_col = st.selectbox(
                "Optional: Color by (categorical)",
                [None] + list(df.select_dtypes(exclude='number').columns)
            )

            if st.button("Generate Scatter Plot"):
                if x_col and y_col:
                    fig, ax = plt.subplots(figsize=(4, 3), dpi=50)
                    sns.scatterplot(
                        data=df,
                        x=x_col,
                        y=y_col,
                        hue=hue_col if hue_col else None,
                        ax=ax
                    )
                    ax.set_title(f"Scatter Plot: {x_col} vs {y_col}", fontsize=10)
                    plt.tight_layout()

                    st.pyplot(fig)
                else:
                    st.warning("Please select both X and Y axes.")

        elif viz_type == "Heatmap":
            numeric_df = df.select_dtypes(include='number')
            if numeric_df.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
                sns.heatmap(
                    numeric_df.corr(numeric_only=True),
                    annot=True,
                    cmap="coolwarm",
                    ax=ax,
                    annot_kws={"size": 8}
                )
                ax.set_title("Correlation Heatmap", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Need at least 2 numeric columns for heatmap.")

    if st.button(f"Hide {viz_type} Section"):
        try:
            st.session_state.hidden_charts[viz_type] = True
            st.experimental_rerun()
        except Exception:
            pass
with v2:
    if viz_type=="Bar Chart":
        st.header("Description of Bar Chart")
        st.write("""A bar chart is a graphical representation of data using rectangular bars, where the length of each bar is proportional to the value it represents. Bar charts are used to compare different categories or groups and visualize discrete data.

                    --The x-axis typically shows the categories.

                    --The y-axis represents the numerical values or frequency.

                    --Bars can be displayed vertically or horizontally.

                    --Useful for showing comparisons among categories or tracking changes over time when categories are ordered."""
                    )
    elif viz_type=="Pie Chart":
        st.header("Description of Pie Chart")
        st.write("""A pie chart is a circular statistical graphic divided into slices to illustrate numerical proportions. Each slice represents a category's contribution to the whole, making it easy to visualize relative sizes.

                    --Each slice's angle is proportional to the quantity it represents.

                    --Best for showing parts of a whole, especially when there are few categories.

                    --Not ideal for comparing similar-sized categories or showing changes over time."""
                    )
    elif viz_type=="Histogram":
        st.header("Description of Histogram")
        st.write("""A histogram is a graphical representation of the distribution of numerical data, showing the frequency of data points within specified ranges (bins).

                    --The x-axis represents the bins or intervals.

                    --The y-axis shows the frequency or count of data points in each bin.

                    --Useful for understanding the distribution, central tendency, and variability of continuous data."""
                    )
    elif viz_type=="Box Plot":
        st.header("Description of Box Plot")
        st.write("""A box plot (or whisker plot) is a standardized way of displaying the distribution of data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum.

                    --The box represents the interquartile range (IQR) between Q1 and Q3.

                    --The line inside the box indicates the median.

                    --Whiskers extend to the minimum and maximum values within 1.5 times the IQR.

                    --Useful for identifying outliers and understanding data spread."""
                    )
    elif viz_type=="Scatter Plot":
        st.header("Description of Scatter Plot")
        st.write("""A scatter plot is a type of data visualization that uses dots to represent the values obtained for two different variables, allowing for the observation of relationships or correlations between them.

                    --The x-axis represents one variable, while the y-axis represents another.

                    --Each point corresponds to an observation in the dataset.

                    --Useful for identifying trends, clusters, and outliers in data."""
                    )
    elif viz_type=="Heatmap":
        st.header("Description of Heatmap")
        st.write("""A heatmap is a data visualization technique that uses color to represent the values of a matrix or table, allowing for quick identification of patterns and correlations.

                    --Each cell's color intensity corresponds to its value.

                    --Useful for visualizing complex data relationships, especially in correlation matrices.

                    --Can highlight areas of high or low values effectively."""
                    )
    else:
        st.write("Select a visualization type to see its description here.")    
                            , structure the code using functions liek opps concept, change the code using oops concept but don't alter the output
