import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from databricks import sql
import plotly.express as px

# Load environment variables
load_dotenv()

# Databricks connection settings
DATABRICKS_SERVER = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_PATH   = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN  = os.getenv("DATABRICKS_ACCESS_TOKEN")

@st.cache_data(ttl=600)
def load_table(query: str) -> pd.DataFrame:
    conn = sql.connect(
        server_hostname=DATABRICKS_SERVER,
        http_path=DATABRICKS_PATH,
        access_token=DATABRICKS_TOKEN
    )
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        cols = [c[0] for c in cursor.description]
        data = cursor.fetchall()
    finally:
        for obj in (cursor, conn):
            try: obj.close()
            except: pass
    return pd.DataFrame(data, columns=cols)

st.set_page_config(page_title="Consumer Goods Analytics", layout="wide")
st.title("📊 Consumer Goods Analytics Demo")

# Create top-level tabs
tabs = st.tabs(["Overview", "Segmentation", "Product Insights"])

# Tab 1: Overview
with tabs[0]:
    st.header("Key Metrics & Forecast")
    df_kpis = load_table("""
      SELECT
        SUM(Total_Amount)            AS total_revenue,
        AVG(Total_Amount)            AS avg_order_value,
        COUNT(DISTINCT Customer_ID)  AS unique_customers
      FROM gold.fact_sales
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue",    f"€{df_kpis.total_revenue[0]:,.0f}")
    c2.metric("Avg Order Value",  f"€{df_kpis.avg_order_value[0]:,.2f}")
    c3.metric("Unique Customers", f"{df_kpis.unique_customers[0]:,}")

    # 30-day forecast chart
    fc = load_table("""
      SELECT ds, yhat, yhat_lower, yhat_upper
      FROM gold.sales_forecast
      ORDER BY ds
    """)
    fig_fc = px.line(
        fc, x="ds", y=["yhat", "yhat_lower", "yhat_upper"],
        labels={"value":"Sales", "ds":"Date"},
        title="30-Day Sales Forecast"
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # TODO: Embed MCP/LLM marketing tips here

# Tab 2: Segmentation
with tabs[1]:
    st.header("Customer Segments")
    seg  = load_table("SELECT * FROM gold.customer_segments")
    cust = load_table("SELECT * FROM gold.dim_customer")
    merged = pd.merge(seg, cust, on="Customer_ID")
    st.dataframe(merged, height=400)

    # Segment size bar chart
    seg_sizes = (
        merged["segment"]
        .value_counts()
        .sort_index()
        .rename_axis("segment")
        .reset_index(name="count")
    )
    fig_seg = px.bar(
        seg_sizes, x="segment", y="count",
        title="Customers per Segment",
        labels={"count":"# Customers","segment":"Segment ID"}
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    # TODO: Embed MCP/LLM segment descriptions here

# Tab 3: Product Insights
with tabs[2]:
    st.header("Top Products & 7-Day Forecast")

    # Top 10 products by revenue (include Product_ID)
    prod = load_table("""
      SELECT Product_ID, Product_Name, SUM(Total_Amount) AS revenue
      FROM gold.fact_sales
      GROUP BY Product_ID, Product_Name
      ORDER BY revenue DESC
      LIMIT 10
    """)
    fig_prod = px.bar(
        prod, x="Product_Name", y="revenue",
        title="Top 10 Products by Revenue"
    )
    st.plotly_chart(fig_prod, use_container_width=True)

    # 7-day forecast for the top product
    top_prod_id   = prod.iloc[0]["Product_ID"]
    top_prod_name = prod.iloc[0]["Product_Name"]
    st.markdown(f"**7-day forecast for:** {top_prod_name}")

    prod_fc = load_table(f"""
      SELECT ds, yhat
      FROM gold.product_forecast
      WHERE Product_ID = '{top_prod_id}'
      ORDER BY ds
    """)
    prod_fc["ds"] = pd.to_datetime(prod_fc["ds"])
    st.line_chart(prod_fc.set_index("ds")["yhat"])

    # TODO: Embed MCP/LLM product tips here
