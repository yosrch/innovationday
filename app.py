import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from databricks import sql
import plotly.express as px
from openai import OpenAI


# Load environment variables
load_dotenv()

# Initialize the new OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2) Debug: list available models *after* client is ready
available = client.models.list()
st.write("Available models:", [m.id for m in available.data])



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

# --- Tab 1: Overview ---
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
        fc,
        x="ds",
        y=["yhat", "yhat_lower", "yhat_upper"],
        labels={"value": "Sales (€)", "ds": "Date"},
        title="30-Day Sales Forecast",
        template="plotly_white"
    )
    fig_fc.for_each_trace(lambda t: t.update(name={
        "yhat": "Forecast",
        "yhat_lower": "Lower Bound",
        "yhat_upper": "Upper Bound"
    }[t.name]))
    fig_fc.update_traces(hovertemplate="%{y:,.0f} €<br>%{x|%Y-%m-%d}")
    fig_fc.update_layout(
        xaxis_title="Date",
        yaxis_title="Forecasted Revenue (€)",
        legend_title="Series"
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # OpenAI-powered marketing tips
    st.subheader("🔍 Automated Insights")
    if st.button("Generate Marketing Tips"):
        prompt = (
            f"Our KPIs are:\n"
            f"- Total Revenue: €{df_kpis.total_revenue[0]:,.0f}\n"
            f"- Avg Order Value: €{df_kpis.avg_order_value[0]:,.2f}\n"
            f"- Unique Customers: {df_kpis.unique_customers[0]}\n\n"
            "Please provide 3 concise, prioritized marketing tips to increase revenue and engagement."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        text = resp.choices[0].message.content
        tips = text.strip().split("\n")
        for tip in tips:
            st.write(f"- {tip}")

# --- Tab 2: Segmentation ---
with tabs[1]:
    st.header("Customer Segments")
    seg  = load_table("SELECT * FROM gold.customer_segments")
    cust = load_table("SELECT * FROM gold.dim_customer")
    merged = pd.merge(seg, cust, on="Customer_ID")
    st.dataframe(merged, height=400)

    seg_sizes = (
        merged["segment"]
        .value_counts()
        .sort_index()
        .rename_axis("segment")
        .reset_index(name="count")
    )
    fig_seg = px.bar(
        seg_sizes,
        x="segment",
        y="count",
        labels={"segment": "Segment ID", "count": "# Customers"},
        title="Customers per Segment",
        template="plotly_white"
    )
    fig_seg.update_traces(hovertemplate="%{y} customers<br>Segment %{x}")
    fig_seg.update_layout(
        xaxis_title="Segment ID",
        yaxis_title="Number of Customers"
    )
    st.plotly_chart(fig_seg, use_container_width=True)

# --- Tab 3: Product Insights ---
with tabs[2]:
    st.header("Top Products & 7-Day Forecast")

    prod = load_table("""
      SELECT Product_ID, Product_Name, SUM(Total_Amount) AS revenue
      FROM gold.fact_sales
      GROUP BY Product_ID, Product_Name
      ORDER BY revenue DESC
      LIMIT 10
    """)
    fig_prod = px.bar(
        prod,
        x="Product_Name",
        y="revenue",
        labels={"Product_Name": "Product", "revenue": "Total Revenue (€)"},
        title="Top 10 Products by Revenue",
        template="plotly_white"
    )
    fig_prod.update_traces(hovertemplate="%{y:,.0f} €<br>%{x}")
    fig_prod.update_layout(
        xaxis_title="Product",
        yaxis_title="Revenue (€)"
    )
    st.plotly_chart(fig_prod, use_container_width=True)

    top_prod_id   = prod.iloc[0]["Product_ID"]
    top_prod_name = prod.iloc[0]["Product_Name"]
    st.markdown(f"**7-Day Sales Forecast for {top_prod_name}**")

    prod_fc = load_table(f"""
      SELECT ds, yhat
      FROM gold.product_forecast
      WHERE Product_ID = '{top_prod_id}'
      ORDER BY ds
    """)
    prod_fc["ds"] = pd.to_datetime(prod_fc["ds"])
    fig_pfc = px.line(
        prod_fc,
        x="ds",
        y="yhat",
        labels={"ds": "Date", "yhat": f"Forecasted Sales (€)"},
        title=f"7-Day Sales Forecast for {top_prod_name}",
        template="plotly_white"
    )
    fig_pfc.update_traces(hovertemplate="%{y:,.0f} €<br>%{x|%Y-%m-%d}", name="Forecast")
    fig_pfc.update_layout(
        xaxis_title="Date",
        yaxis_title="Forecasted Sales (€)"
    )
    st.plotly_chart(fig_pfc, use_container_width=True)

