#main
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from databricks import sql
import plotly.express as px

# Load environment variables from .env (Streamlit Cloud will ignore this file in deployment)
load_dotenv()

# Databricks connection settings from env
DATABRICKS_SERVER = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_PATH   = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN  = os.getenv("DATABRICKS_ACCESS_TOKEN")

@st.cache_data(ttl=600)
def load_table(query: str) -> pd.DataFrame:
    # Open connection
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
        # Attempt to close cursor & connection, ignoring any errors
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    return pd.DataFrame(data, columns=cols)

st.set_page_config(page_title="Consumer Goods Analytics", layout="wide")
st.title("📊 Consumer Goods Analytics Demo")

# --- Create top-level tabs ---
tabs = st.tabs(["Overview", "Segmentation", "Product Insights"])

# --- Tab 1: Overview ---
with tabs[0]:
    st.header("Key Metrics & Forecast")
    df_kpis = load_table("""
      SELECT
        SUM(Total_Amount)  AS total_revenue,
        AVG(Total_Amount)  AS avg_order_value,
        COUNT(DISTINCT Customer_ID) AS unique_customers
      FROM gold.fact_sales
    """)
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"€{df_kpis.total_revenue[0]:,.0f}")
    col2.metric("Avg Order Value", f"€{df_kpis.avg_order_value[0]:,.2f}")
    col3.metric("Unique Customers", f"{df_kpis.unique_customers[0]:,}")

    # 30-day forecast
    fc = load_table("SELECT ds, yhat, yhat_lower, yhat_upper FROM gold.sales_forecast ORDER BY ds")
    fig_fc = px.line(fc, x="ds", y=["yhat", "yhat_lower", "yhat_upper"],
                     labels={"value":"Sales", "ds":"Date"},
                     title="30-Day Sales Forecast")
    st.plotly_chart(fig_fc, use_container_width=True)

    # Placeholder: MCP/LLM generate natural language summary & tips
    st.subheader("🔍 Automated Insights")
    if st.button("Generate Marketing Tips"):
        prompt = (
            f"Our KPIs are:\n"
            f"- Revenue: €{df_kpis.total_revenue[0]:,.0f}\n"
            f"- AOV: €{df_kpis.avg_order_value[0]:,.2f}\n"
            f"- Customers: {df_kpis.unique_customers[0]}\n\n"
            "Give me 3 actionable marketing tips."
        )
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7,
            max_tokens=150
        )
        tips = resp.choices[0].message.content.strip().split("\n")
        for tip in tips:
            st.write(f"- {tip}")

# --- Tab 2: Segmentation ---
with tabs[1]:
    st.header("Customer Segments")
    seg   = load_table("SELECT * FROM gold.customer_segments")
    cust  = load_table("SELECT * FROM gold.dim_customer")
    merged = pd.merge(seg, cust, on="Customer_ID")
    st.dataframe(merged, height=400)

    # Bar chart of segment sizes
    seg_sizes = merged["segment"].value_counts().sort_index().reset_index()
    seg_sizes.columns = ["segment", "count"]
    fig_seg = px.bar(seg_sizes, x="segment", y="count",
                     title="Customers per Segment",
                     labels={"count":"# Customers","segment":"Segment ID"})
    st.plotly_chart(fig_seg, use_container_width=True)

    # Placeholder: LLM describe segment characteristics
    if st.button("Describe Segments"):
        prompt = "Describe the characteristics of each customer segment based on the data provided."
        # ... call openai here and display ...

# --- Tab 3: Product Insights ---
with tabs[2]:
    st.header("Top Products & Forecast")
    prod = load_table("""
      SELECT Product_Name, SUM(Total_Amount) AS revenue
      FROM gold.fact_sales
      GROUP BY  Product_ID, Product_Name
      ORDER BY revenue DESC
      LIMIT 10
    """)
    fig_prod = px.bar(prod, x="Product_Name", y="revenue",
                      title="Top 10 Products by Revenue")
    st.plotly_chart(fig_prod, use_container_width=True)

    # Product-level 7-day forecast example for top product
    # … inside with tabs[2]: …

# 7-day forecast example for top product (using Product_ID)
top_prod_id = prod.iloc[0]["Product_ID"]
top_prod_name = prod.iloc[0]["Product_Name"]
st.markdown(f"**7-day forecast for:** {top_prod_name}")

prod_fc = load_table(f"""
  SELECT ds, yhat
  FROM gold.product_forecast
  WHERE Product_ID = '{top_prod_id}'
  ORDER BY ds
""")
prod_fc["ds"] = pd.to_datetime(prod_fc["ds"])  # ensure correct dtype
st.line_chart(prod_fc.set_index("ds")["yhat"])


# Placeholder: LLM suggest cross-sell or pricing tips
if st.button("Suggest Product Tips"):
    prompt = f"For the product {top_prod}, suggest pricing or cross-sell strategies."
        # ... call openai here and display ...