import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from databricks import sql
import plotly.express as px
import requests

st.markdown(
    """
    <style>
      /* Add some breathing room around charts and tables */
      .stPlotlyChart, .streamlit-expanderHeader {
        margin-top: 1rem;
        margin-bottom: 1rem;
      }
      /* Style LLM answer boxes */
      .llm-box {
        background-color: #f0f4ff;
        border-left: 4px solid #3f51b5;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)
# Load environment variables
load_dotenv()

# Initialize CLAUDE client
CLAUDE_URL   = os.getenv("CLAUDE_ENDPOINT_URL")
CLAUDE_TOKEN = os.getenv("CLAUDE_BEARER_TOKEN")

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
            try:
                obj.close()
            except:
                pass
    return pd.DataFrame(data, columns=cols)

st.set_page_config(page_title="Consumer Goods Analytics", layout="wide")
st.title("📊 Consumer Goods Analytics Demo")

# Create top-level tabs
tabs = st.tabs(["Overview", "Segmentation", "Product Insights", "Ask the Data"])

# --- Tab 1: Overview ---
with tabs[0]:
    st.subheader("Key Metrics & Forecast")
    df_kpis = load_table("""
      SELECT
        SUM(Total_Amount)            AS total_revenue,
        AVG(Total_Amount)            AS avg_order_value,
        COUNT(DISTINCT Customer_ID)  AS unique_customers
      FROM gold.fact_sales
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("💰Total Revenue",    f"€{df_kpis.total_revenue[0]:,.0f}")
    c2.metric("📈 Avg Order Value",  f"€{df_kpis.avg_order_value[0]:,.2f}")
    c3.metric("👥Unique Customers", f"{df_kpis.unique_customers[0]:,}")

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
    st.subheader("30-Day Sales Forecast")
    st.plotly_chart(fig_fc, use_container_width=True)

    # OpenAI-powered marketing tips
with tabs[0]:
    st.subheader("Key Metrics & Forecast")

    # — Metrics row —
    with st.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Revenue",    f"€{df_kpis.total_revenue[0]:,.0f}")
        c2.metric("📈 Avg Order Value",  f"€{df_kpis.avg_order_value[0]:,.2f}")
        c3.metric("👥 Unique Customers", f"{df_kpis.unique_customers[0]:,}")

    # — Forecast chart —
    st.subheader("30-Day Sales Forecast")
    st.plotly_chart(fig_fc, use_container_width=True)

    # — AI tips in an expander —
    with st.expander("🔍 Automated Marketing Tips", expanded=False):
        if st.button("Generate General Tips"):
            prompt = (
                f"Our KPIs are:\n"
                f"- Total Revenue: €{df_kpis.total_revenue[0]:,.0f}\n"
                f"- Avg Order Value: €{df_kpis.avg_order_value[0]:,.2f}\n"
                f"- Unique Customers: {df_kpis.unique_customers[0]}\n\n"
                "Please provide 3 concise, prioritized marketing tips to increase revenue and engagement."
            )
            headers = {
                "Authorization": f"Bearer {CLAUDE_TOKEN}",
                "Content-Type": "application/json"
            }
            body = {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            try:
                r = requests.post(CLAUDE_URL, json=body, headers=headers, timeout=120)
                if r.status_code != 200:
                    st.error(f"Invocation failed with status {r.status_code}")
                    st.code(r.text, language="json")
                    st.stop()
                resp_json = r.json()
                text = resp_json["choices"][0]["message"]["content"]
            except Exception as e:
                st.error("Failed to generate tips. Please try again later.")
                st.exception(e)
                st.stop()

            # Render each tip in a styled box
            tips = [line.strip() for line in text.splitlines() if line.strip()]
            for tip in tips:
                st.markdown(f"<div class='llm-box'>• {tip}</div>", unsafe_allow_html=True)

# --- Tab 2: Segmentation ---
with tabs[1]:
    st.subheader("Customer Segments Overview")

    # — Table & chart side-by-side —
    left, right = st.columns([2, 1])
    with left:
        st.dataframe(merged, height=300)
    with right:
        st.plotly_chart(fig_seg, use_container_width=True)

    # Compute segment statistics with purchase behavior
    seg_stats = load_table("""
      SELECT
        s.segment,
        COUNT(DISTINCT c.Customer_ID)       AS count,
        ROUND(AVG(c.Age),1)                 AS avg_age,
        ROUND(100.0*SUM(CASE WHEN c.Gender='Male' THEN 1 ELSE 0 END)/COUNT(DISTINCT c.Customer_ID),1)
                                            AS pct_male,
        ROUND(AVG(f.Total_Amount),2)        AS avg_order_value,
        ROUND(COUNT(f.Customer_ID)/COUNT(DISTINCT c.Customer_ID),2)
                                            AS avg_orders_per_customer
      FROM gold.customer_segments s
      JOIN gold.dim_customer c
        ON s.Customer_ID = c.Customer_ID
      JOIN gold.fact_sales f
        ON c.Customer_ID = f.Customer_ID
      GROUP BY s.segment
      ORDER BY s.segment
    """)
    total_customers = seg_stats["count"].sum()

    # — Profiles expander —
    with st.expander("👥 Describe Customer Segments", expanded=False):
        if st.button("Describe Segments"):
            lines = []
            for _, r in seg_stats.iterrows():
                pct = r["count"] / total_customers * 100
                lines.append(
                    f"Segment {int(r['segment'])}: {int(r['count'])} customers "
                    f"({pct:.1f}%), Avg age {r['avg_age']}, {r['pct_male']:.1f}% male, "
                    f"{r['avg_orders_per_customer']:.2f} orders/customer, "
                    f"Avg order €{r['avg_order_value']:.2f}"
                )
            prompt = "Here are our customer segments:\n" + "\n".join(f"- {l}" for l in lines) + \
                     "\n\nFor each segment, write a 1–2 sentence profile describing its key characteristics."

            body = {"messages": [{"role":"user","content":prompt}]}
            with st.spinner("Generating segment profiles…"):
                r = requests.post(CLAUDE_URL, json=body, headers={"Authorization":f"Bearer {CLAUDE_TOKEN}","Content-Type":"application/json"}, timeout=120)
                if r.status_code != 200:
                    st.error(f"Invocation failed: {r.status_code}")
                    st.code(r.text, language="json")
                    st.stop()
                msg = r.json()["choices"][0]["message"]["content"]

            for line in [l.strip() for l in msg.splitlines() if l.strip()]:
                st.markdown(f"<div class='llm-box'>{line}</div>", unsafe_allow_html=True)

    # — Strategies expander —
    with st.expander("🎯 Segment-Specific Strategies", expanded=False):
        if st.button("Generate Segment Strategies"):
            prompt = "We have the following customer segments:\n" + \
                     "\n".join(f"- Segment {int(r['segment'])}: {int(r['count'])} customers, "
                               f"{r['avg_orders_per_customer']:.2f} orders/customer, "
                               f"Avg order €{r['avg_order_value']:.2f}"
                               for _, r in seg_stats.iterrows()) + \
                     "\n\nFor each segment, recommend its top marketing channel, an offer type (discount, bundle, free shipping), " \
                     "and which ABC product category to emphasize. Give 2 bullet points per segment."

            body = {"messages":[{"role":"user","content":prompt}]}
            with st.spinner("Generating segment strategies…"):
                r = requests.post(CLAUDE_URL, json=body, headers={"Authorization":f"Bearer {CLAUDE_TOKEN}","Content-Type":"application/json"}, timeout=120)
                if r.status_code != 200:
                    st.error(f"Invocation failed: {r.status_code}")
                    st.code(r.text, language="json")
                    st.stop()
                msg = r.json()["choices"][0]["message"]["content"]

            for line in [l.strip() for l in msg.splitlines() if l.strip()]:
                st.markdown(f"<div class='llm-box'>{line}</div>", unsafe_allow_html=True)

# --- Tab 3: Product Insights ---
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

    # 7-day forecast for the top product
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


    # 1) Load ABC classifications
    prod_abc = load_table("SELECT * FROM gold.product_abc ORDER BY revenue DESC")

    st.subheader("📦 ABC Classification of Top Products")

    # 2) Create two columns: left for grid, right for treemap
    grid_col, tree_col = st.columns([2, 1])

    # 3) In the left column, show AgGrid
    with grid_col:
        from st_aggrid import AgGrid, GridOptionsBuilder

        gb = GridOptionsBuilder.from_dataframe(prod_abc)
        gb.configure_default_column(filterable=True, sortable=True, resizable=True)
        grid_opts = gb.build()

        AgGrid(
            prod_abc,
            gridOptions=grid_opts,
            enable_enterprise_modules=False,
            theme="alpine",
            height=400
        )

    # 4) In the right column, show a Plotly treemap
    with tree_col:
        import plotly.express as px

        fig_treemap = px.treemap(
            prod_abc,
            path=["ABC_Category", "Product_Name"],
            values="revenue",
            color="ABC_Category",
            color_discrete_map={"A":"gold","B":"lightblue","C":"lightgray"},
            title="Revenue by ABC Category"
        )
        fig_treemap.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_treemap, use_container_width=True)

# If you want to let Claude suggest strategies per category:
    if st.button("Generate ABC-Based Product Strategies"):
        # Build the prompt from prod_abc DataFrame
        prompt = (
            "We have these product categories based on ABC analysis:\n"
            + "\n".join(
                f"- {row['Product_Name']}: Category {row['ABC_Category']}, Revenue €{row['revenue']:,}"
                for _, row in prod_abc.iterrows()   # <-- unpack index, row
            )
            + "\n\nFor each category (A, B, C), recommend pricing or promotional strategies."
        )

        headers = {
            "Authorization": f"Bearer {CLAUDE_TOKEN}",
            "Content-Type": "application/json"
        }
        body = {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        # Show a spinner while waiting
        with st.spinner("Generating strategies (this may take up to 2 minutes)…"):
            try:
                r = requests.post(
                    CLAUDE_URL,
                    json=body,
                    headers=headers,
                    timeout=120  # <-- give it up to two minutes
                )
            except requests.exceptions.ReadTimeout:
                st.error("The request timed out (took over 2 minutes). Try again or switch to a lighter model.")
                st.stop()
            except Exception as e:
                st.error("Failed to generate product strategies via Claude.")
                st.exception(e)
                st.stop()

        # Now check the status code
        if r.status_code != 200:
            st.error(f"Invocation failed with status {r.status_code}")
            st.code(r.text, language="json")
            st.stop()

        resp_json = r.json()
        text = resp_json["choices"][0]["message"]["content"]

        for line in text.split("\n"):
            if line.strip():
                st.write(f"- {line.strip()}")



@st.cache_data(ttl=600)
def get_data_context() -> str:
    # 1) KPIs
    df_kpis = load_table("""
      SELECT
        SUM(Total_Amount)  AS total_revenue,
        AVG(Total_Amount)  AS avg_order_value,
        COUNT(DISTINCT Customer_ID) AS unique_customers
      FROM gold.fact_sales
    """)
    # 2) Segments
    seg_sizes = load_table("""
      SELECT segment, COUNT(*) AS count
      FROM gold.customer_segments
      GROUP BY segment
      ORDER BY segment
    """)
    # 3) ABC categories
    prod_abc = load_table("SELECT Product_Name, ABC_Category FROM gold.product_abc")

    # Build a single string
    context = (
        f"KPI: Revenue €{df_kpis.total_revenue[0]:,.0f}, "
        f"AOV €{df_kpis.avg_order_value[0]:,.2f}, "
        f"Customers {df_kpis.unique_customers[0]:,}\n\n"
        "Segments:\n"
        + "\n".join(f"- {int(r.segment)}: {int(r['count']):,} customers"
                    for _, r in seg_sizes.iterrows())
        + "\n\nProduct ABC Categories:\n"
        + ", ".join(f"{row.Product_Name}({row.ABC_Category})"
                    for _, row in prod_abc.iterrows())
    )
    return context

# --- Tab 4: Ask the Data ---
with tabs[3]:
    st.subheader("💬 Ask the Data")

    # — Chat container to hold all messages —
    chat_container = st.container()

    # — Initialize chat history if needed —
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # — Helper to (re)draw just the user & assistant messages —
    def render_history():
        chat_container.empty()
        for msg in st.session_state.messages:
            chat_container.chat_message(msg["role"]).write(msg["content"])

    # — Render any existing history on load —
    render_history()

    # — Input box for a new question —
    user_question = st.chat_input("Type your question about KPIs, segments or products…")
    if user_question:
        # 1) Record & immediately display the user’s message
        st.session_state.messages.append({"role": "user", "content": user_question})
        chat_container.chat_message("user").write(user_question)

        # 2) Build the prompt context (cached)
        data_context = get_data_context()
        prompt = (
            f"Context:\n{data_context}\n\n"
            f"Question: {user_question}"
        )

        # 3) Prepare the three-message Claude payload
        body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert data analyst assistant. "
                        "Answer concisely in bullet points without repeating full context."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Question: What is our current average order value?\n"
                        "Answer: The average order value is €1,284, reflecting strong upsell performance."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        headers = {
            "Authorization": f"Bearer {CLAUDE_TOKEN}",
            "Content-Type": "application/json",
        }

        # 4) Call the Claude endpoint
        with st.spinner("Thinking…"):
            r = requests.post(CLAUDE_URL, json=body, headers=headers, timeout=120)
            if r.status_code != 200:
                st.error(f"Invocation failed: {r.status_code}")
                st.code(r.text, language="json")
                st.stop()
            assistant_reply = r.json()["choices"][0]["message"]["content"]

        # 5) Strip out any <<…>> tokens
        cleaned_lines = [
            line for line in assistant_reply.splitlines()
            if not (line.strip().startswith("<<") and line.strip().endswith(">>"))
        ]
        assistant_reply = "\n".join(cleaned_lines).strip()

        # 6) Record & immediately display the assistant’s reply
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        chat_container.chat_message("assistant").write(assistant_reply)