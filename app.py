import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from databricks import sql

# Load environment variables from .env (Streamlit Cloud will ignore this file in deployment)
load_dotenv()

# Databricks connection settings from env
DATABRICKS_SERVER = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_PATH   = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN  = os.getenv("DATABRICKS_ACCESS_TOKEN")

@st.cache_data(ttl=600)
def load_table(query: str) -> pd.DataFrame:
    with sql.connect(
        server_hostname=DATABRICKS_SERVER,
        http_path=DATABRICKS_PATH,
        access_token=DATABRICKS_TOKEN
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            cols = [c[0] for c in cursor.description]
            data = cursor.fetchall()
    return pd.DataFrame(data, columns=cols)

st.set_page_config(page_title="Demo Analytics", layout="wide")
st.title("📊 Consumer Goods Analytics Demo")

# Sidebar navigation
page = st.sidebar.selectbox("Choose View", ["Overview", "Segmentation", "Product Insights"])

if page == "Overview":
    st.header("Key Metrics")
    # You can create a SQL view gold.kpi_view in Databricks or compute here:
    df_kpis = load_table("""
        SELECT
          SUM(Total_Amount) AS total_revenue,
          AVG(Total_Amount) AS avg_order_value,
          COUNT(DISTINCT Customer_ID) AS unique_customers
        FROM gold.fact_sales
    """)
    st.metric("Total Revenue", f"€{df_kpis.total_revenue[0]:,.2f}")
    st.metric("Avg Order Value", f"€{df_kpis.avg_order_value[0]:,.2f}")
    st.metric("Unique Customers", df_kpis.unique_customers[0])

elif page == "Segmentation":
    st.header("Customer Segments")
    seg_df = load_table("SELECT * FROM gold.customer_segments")
    cust_df = load_table("SELECT * FROM gold.dim_customer")
    merged = pd.merge(seg_df, cust_df, on="Customer_ID")
    st.dataframe(merged)

else:  # Product Insights
    st.header("Top Products by Revenue")
    prod_df = load_table("""
        SELECT Product_Name, SUM(Total_Amount) AS Revenue
        FROM gold.fact_sales
        GROUP BY Product_Name
        ORDER BY Revenue DESC
        LIMIT 10
    """)
    st.bar_chart(prod_df.set_index("Product_Name")["Revenue"])
