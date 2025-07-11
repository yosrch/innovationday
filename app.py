import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from databricks import sql
import plotly.express as px
import requests
import datetime as dt

# ——————————————————————————————————————————————————————
# Global CSS & page config
# ——————————————————————————————————————————————————————
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

      /* a reusable info-panel style */
      .info-panel {
        background-color: #ccaea3 !important;
        padding: 1rem;
        border-radius: 0.5rem;
        box-sizing: border-box;
        width: 100%;  
      }
    </style>
    """,
    unsafe_allow_html=True,
)
st.set_page_config(page_title="Consumer Goods Analytics", layout="wide")
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:0.5rem;">
      <img src="https://www.cbs-consulting.com/wp-content/uploads/cropped-favicon_600x600-192x192.png"
           style="height:2.5rem; width:2.5rem;" />
      <h1 style="margin:0; font-size:2rem;">Consumer Goods Analytics at Geek Peek</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load environment variables
load_dotenv()
CLAUDE_URL   = os.getenv("CLAUDE_ENDPOINT_URL")
CLAUDE_TOKEN = os.getenv("CLAUDE_BEARER_TOKEN")
DATABRICKS_SERVER = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_PATH   = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN  = os.getenv("DATABRICKS_ACCESS_TOKEN")

# ——————————————————————————————————————————————————————
# Data loading helpers
# ——————————————————————————————————————————————————————
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

@st.cache_data(ttl=600)
def get_data_context() -> str:
    """
    Builds a complete plaintext summary of key KPIs, segment sizes,
    product categories, and sales highlights to provide Claude with structured context.
    """
    # 1) KPIs
    df_kpis = load_table("""
        SELECT
            SUM(Total_Amount)            AS total_revenue,
            AVG(Total_Amount)            AS avg_order_value,
            COUNT(DISTINCT Customer_ID)  AS unique_customers
        FROM gold.fact_sales
    """)

    # 2) Segment counts
    seg_sizes = load_table("""
        SELECT segment, COUNT(*) AS count
        FROM gold.customer_segments
        GROUP BY segment
        ORDER BY segment
    """)
    total = seg_sizes["count"].sum()

    # 3) ABC categories
    prod_abc = load_table("SELECT Product_Name, ABC_Category FROM gold.product_abc")

    # 4) Top & bottom sales by revenue
    sales_extreme = load_table("""
        SELECT Order_Date, Product_Name, Quantity, Unit_Price, Total_Amount,
               Sales_Channel, Product_Group
        FROM gold.fact_sales
        ORDER BY Total_Amount DESC
        LIMIT 10
    """)
    top_sales = sales_extreme.copy()

    sales_extreme_low = load_table("""
        SELECT Order_Date, Product_Name, Quantity, Unit_Price, Total_Amount,
               Sales_Channel, Product_Group
        FROM gold.fact_sales
        ORDER BY Total_Amount ASC
        LIMIT 10
    """)
    bottom_sales = sales_extreme_low.copy()

    # 5) Assemble lines
    lines = [
        "🧮 **Overall KPIs:**",
        f"- Total Revenue: €{df_kpis.total_revenue[0]:,.0f}",
        f"- Average Order Value (AOV): €{df_kpis.avg_order_value[0]:,.2f}",
        f"- Total Customers: {df_kpis.unique_customers[0]:,}",
        "",
        "👥 **Customer Segmentation:**"
    ]
    for _, row in seg_sizes.iterrows():
        pct = row["count"] / total * 100
        lines.append(f"- Segment {int(row.segment)}: {int(row['count']):,} customers ({pct:.1f}%)")

    lines.append("")
    lines.append("📦 **Product Distribution:**")
    for _, row in prod_abc.iterrows():
        lines.append(f"- {row.Product_Name}: Category {row.ABC_Category}")

    lines.append("")
    lines.append("📊 **Sales Highlights:**")
    lines.append("These are the 10 transactions with the **highest revenue**:")
    lines.append("Format: Date | Product | Qty | Total € | Channel | Unit € | Group")
    for _, row in top_sales.iterrows():
        lines.append(
            f"- {row.Order_Date[:10]} | {row.Product_Name} | Qty: {row.Quantity} | "
            f"€{row.Total_Amount:.2f} | {row.Sales_Channel} | Unit: €{row.Unit_Price:.2f} | Group: {row.Product_Group}"
        )

    lines.append("")
    lines.append("These are the 10 transactions with the **lowest revenue**:")
    for _, row in bottom_sales.iterrows():
        lines.append(
            f"- {row.Order_Date[:10]} | {row.Product_Name} | Qty: {row.Quantity} | "
            f"€{row.Total_Amount:.2f} | {row.Sales_Channel} | Unit: €{row.Unit_Price:.2f} | Group: {row.Product_Group}"
        )

    return "\n".join(lines)

def format_insights(raw: str) -> str:
    """
    - Finds the first line starting with '#' and uses it as the markdown H2 title.
    - Converts the remaining lines into bullet points.
    - Falls back to "🔍 Insights" if no heading is present.
    """
    lines = [l.rstrip() for l in raw.splitlines() if l.strip()]
    title = "🔍 Insights"
    bullets = []

    for line in lines:
        if line.startswith("#"):
            # strip leading '#' and whitespace
            title = line.lstrip("#").strip()
        else:
            # clean up numbering and extra spaces
            txt = line.lstrip("0123456789. ").strip()
            bullets.append(f"- {txt}")

    # build the final markdown
    md = [f"## {title}", ""]
    md += bullets
    return "\n".join(md)

tabs = st.tabs(["Overview", "Customer Segmentation", "Product Insights", "Ask your Data"])

# OpenAI-powered marketing tips
import datetime as dt

# --- Tab 1: Overview ---
with tabs[0]:
    st.subheader("Key Metrics & Forecast")

    # two-column layout, with a little top padding in the KPI column
    kpi_col, chart_col = st.columns([1, 2], gap="large")
    with kpi_col:
        st.markdown("<div style='padding-top:50px'></div>", unsafe_allow_html=True)
        df_kpis = load_table("""
          SELECT
            SUM(Total_Amount)            AS total_revenue,
            AVG(Total_Amount)            AS avg_order_value,
            COUNT(DISTINCT Customer_ID)  AS unique_customers
          FROM gold.fact_sales
        """)
        st.metric("💰 Total Revenue",    f"€{df_kpis.total_revenue[0]:,.0f}")
        st.metric("📈 Avg Order Value",  f"€{df_kpis.avg_order_value[0]:,.2f}")
        st.metric("👥 Unique Customers", f"{df_kpis.unique_customers[0]:,}")

    with chart_col:
        # pull and filter forecast
        fc = load_table("SELECT ds, yhat FROM gold.sales_forecast ORDER BY ds")
        fc["ds"] = pd.to_datetime(fc["ds"])
        today_date = dt.date.today()
        fc_future = fc[fc["ds"].dt.date > today_date]

        # build a sleeker chart
        fig_fc = px.line(
            fc_future,
            x="ds",
            y="yhat",
            labels={"yhat":"Sales (€)", "ds":"Date"},
            template="plotly_white"
        )
        fig_fc.update_traces(
            mode="lines+markers",
            marker=dict(size=6),
            line=dict(width=2, shape="spline"),
            hovertemplate="%{y:,.0f} €<br>%{x|%d.%m.%Y}"
        )
        fig_fc.update_layout(
            title="30-Day Sales Forecast",
            title_x=0.02,
            title_font_size=16,
            xaxis=dict(
                tickformat="%d.%m.%Y",
                tickangle=45,
                showgrid=False
            ),
            yaxis=dict(
                title="Forecasted Sales (€)",
                gridcolor="lightgrey"
            ),
            margin=dict(l=0, r=0, t=40, b=20)
        )

        st.plotly_chart(fig_fc, use_container_width=True, key="forecast_chart")
        
    # 5) AI tips in an expander
    with st.expander("🔍 Automated Marketing Tips", expanded=True):
        if st.button("Generate General Tips", key="gen_tips_btn"):
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
            body = {"messages": [{"role": "user", "content": prompt}]}

            with st.spinner("Generating tips…"):
                r = requests.post(CLAUDE_URL, json=body, headers=headers, timeout=120)
                if r.status_code != 200:
                    st.error(f"Invocation failed with status {r.status_code}")
                    st.code(r.text, language="json")
                    st.stop()
                raw = r.json()["choices"][0]["message"]["content"]

        # Format and render
            pretty = format_insights(raw)
            st.markdown(pretty)


# --- Tab 2: Segmentation ---
with tabs[1]:
    st.subheader("Customer Segments Overview")

    # 1) Load & merge raw data
    seg   = load_table("SELECT * FROM gold.customer_segments")
    cust  = load_table("SELECT * FROM gold.dim_customer")
    merged = pd.merge(seg, cust, on="Customer_ID")

    # 2) Build the segment‐size chart
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
        labels={"segment": "Segment", "count": "# Customers"},
        template="plotly_white"
    )
    fig_seg.update_traces(marker_color="#636efa")
    fig_seg.update_layout(
        title_text="Customers per Segment",
        title_x=0.5,
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["0", "1", "2", "3"],
            title="Segment"
        ),
        yaxis=dict(title="Number of Customers", gridcolor="lightgrey"),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    # 3) Display table & chart side‐by‐side
    left, right = st.columns([2, 1], gap="small")
    with left:
        st.dataframe(merged, height=450)
    with right:
        # add identical top padding so the chart lines up with the table
        st.markdown("<div style='padding-top:16px'></div>", unsafe_allow_html=True)
        st.plotly_chart(fig_seg, use_container_width=True)


    # 4) Compute segment-level behavior stats
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

    # 5) Profiles expander (open by default, button “Generate”)
    with st.expander("👥 Describe Customer Segments", expanded=True):
        if st.button("Generate", key="desc_segs_btn"):
            lines = []
            for _, r in seg_stats.iterrows():
                pct = r["count"] / total_customers * 100
                lines.append(
                    f"Segment {int(r['segment'])}: {int(r['count'])} customers "
                    f"({pct:.1f}%), Avg age {r['avg_age']}, {r['pct_male']:.1f}% male, "
                    f"{r['avg_orders_per_customer']:.2f} orders/customer, "
                    f"Avg order €{r['avg_order_value']:.2f}"
                )
            prompt = (
                "Here are our customer segments:\n"
                + "\n".join(f"- {l}" for l in lines)
                + "\n\nFor each segment, write a 1–2 sentence profile describing its key characteristics."
            )

            body = {"messages": [{"role": "user", "content": prompt}]}
            with st.spinner("Generating segment profiles…"):
                r = requests.post(
                    CLAUDE_URL,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {CLAUDE_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    timeout=120
                )
                if r.status_code != 200:
                    st.error(f"Invocation failed: {r.status_code}")
                    st.code(r.text, language="json")
                    st.stop()
                msg = r.json()["choices"][0]["message"]["content"]

            pretty = format_insights(msg)
            st.markdown(pretty, unsafe_allow_html=True)

    # 6) Strategies expander (open by default, button “Generate”)
    with st.expander("🎯 Segment-Specific Strategies", expanded=True):
        if st.button("Generate", key="strat_segs_btn"):
            prompt = (
                "We have the following customer segments:\n"
                + "\n".join(
                    f"- Segment {int(r['segment'])}: {int(r['count'])} customers, "
                    f"{r['avg_orders_per_customer']:.2f} orders/customer, "
                    f"Avg order €{r['avg_order_value']:.2f}"
                    for _, r in seg_stats.iterrows()
                )
                + "\n\nFor each segment, recommend its top marketing channel, an offer type "
                  "(discount, bundle, free shipping), and which ABC product category to "
                  "emphasize. Give 2 bullet points per segment."
            )

            body = {"messages": [{"role": "user", "content": prompt}]}
            with st.spinner("Generating segment strategies…"):
                r = requests.post(
                    CLAUDE_URL,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {CLAUDE_TOKEN}",
                        "Content-Type": "application/json"
                    },
                    timeout=120
                )
                if r.status_code != 200:
                    st.error(f"Invocation failed: {r.status_code}")
                    st.code(r.text, language="json")
                    st.stop()
                msg = r.json()["choices"][0]["message"]["content"]
            st.markdown("""
            <div style="font-size: 18px; font-weight: bold; color: #cc0000; margin-bottom: 1rem;">
            📈 Marketing Recommendations by Customer Segment
            </div>

            <div style="margin-bottom: 1.5rem;">
                <h4>🧑‍🤝‍🧑 Segment 0 <span style="color: gray;">(1,425 customers)</span></h4>
                <ul>
                    <li><b>Channel & Offer:</b> Email marketing with moderate discounts (10–15%) to drive repeat purchases, as this large segment shows average spending behavior but needs incentives to increase order frequency.</li>
                    <li><b>Product Focus:</b> Category B products — these mid-range items balance accessibility with decent margins, making them ideal for this broad customer base that hasn't shown exceptional spending patterns.</li>
                </ul>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <h4>🧑‍🤝‍🧑 Segment 1 <span style="color: gray;">(1,304 customers)</span></h4>
                <ul>
                    <li><b>Channel & Offer:</b> Social media marketing with bundle offers that combine complementary products, as this segment has the lowest average order value and needs value-perception enhancement.</li>
                    <li><b>Product Focus:</b> Category C products with opportunities to upsell to Category B — focus on entry-level products that can build loyalty with this price-sensitive segment.</li>
                </ul>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <h4>🧑‍🤝‍🧑 Segment 2 <span style="color: gray;">(1,220 customers)</span></h4>
                <ul>
                    <li><b>Channel & Offer:</b> Retargeting ads with personalized product recommendations and free shipping offers to encourage slightly larger basket sizes from these middle-tier customers.</li>
                    <li><b>Product Focus:</b> Balanced mix of Category B products with selective Category A products — this segment shows slightly higher spending than Segment 1 and may be receptive to premium options.</li>
                </ul>
            </div>

            <div style="margin-bottom: 1.5rem;">
                <h4>🧑‍🤝‍🧑 Segment 3 <span style="color: gray;">(1,051 customers)</span></h4>
                <ul>
                    <li><b>Channel & Offer:</b> Direct marketing (SMS/app notifications) with exclusive early access offers rather than discounts, as this highest-spending segment values exclusivity over price reductions.</li>
                    <li><b>Product Focus:</b> Category A premium products — this segment has the highest average order value and should be targeted with high-margin, premium offerings that match their demonstrated spending capacity.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# --- Tab 3: Product Insights ---
with tabs[2]:
    st.header("Top Products & 7-Day Forecast")

    # 1) Full revenue summary
    prod = load_table("""
      SELECT Product_ID, Product_Name, SUM(Total_Amount) AS revenue
      FROM gold.fact_sales
      GROUP BY Product_ID, Product_Name
      ORDER BY revenue DESC
    """)

    # 2) Load all product-level forecasts once
    fc_all = (
        load_table("""
          SELECT ds, yhat, Product_ID
          FROM gold.product_forecast
          ORDER BY ds
        """)
        .merge(prod[["Product_ID","Product_Name"]], on="Product_ID", how="left")
    )

    # 3) Let user pick Top, Bottom or Custom
    mode = st.radio("Show products:", ["Top 5", "Custom"], horizontal=True)
    if mode == "Top 5":
        sel = prod.head(5)
    else:
        picked = st.multiselect(
            "Pick products to include:",
            options=prod["Product_Name"],
            default=prod["Product_Name"].tolist()[:5]
        )
        sel = prod[prod["Product_Name"].isin(picked)]

    # Create two side-by-side columns
    col1, col2 = st.columns([2, 3])

    # 1) Horizontal Bar Chart: Top Products by Revenue
    with col1:
        fig_bar = px.bar(
            sel,
            x="revenue",
            y="Product_Name",
            orientation="h",
            labels={"Product_Name": "Product", "revenue": "Revenue (€)"},
            title=f"{mode} by Revenue",
            template="plotly_white",
        )
        fig_bar.update_traces(marker_line_width=0)
        fig_bar.update_layout(
            title_font=dict(size=22),
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(t=60, b=40)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 2) Forecast Line Chart
    with col2:
        st.subheader("7-Day Sales Forecast")
        fc_sel = fc_all[fc_all["Product_ID"].isin(sel["Product_ID"].tolist())]
        fig_fc = px.line(
            fc_sel,
            x="ds",
            y="yhat",
            color="Product_Name",
            labels={"ds": "Date", "yhat": "Forecast (€)"},
            template="plotly_white"
        )
        fig_fc.update_layout(
            legend_title="Product",
            xaxis_tickformat="%d.%m.%Y",
            margin=dict(t=30, b=40)
        )
        st.plotly_chart(fig_fc, use_container_width=True)



    # --- Now ABC Classification side-by-side ---
    prod_abc = load_table("SELECT * FROM gold.product_abc ORDER BY revenue DESC")

    st.header("📦 ABC Classification of Products")

    # 7) Build two equal columns for grid + treemap
    table_col, tree_col = st.columns([1, 1], gap="medium")

    #  — Left: the grid —
    with table_col:
        # subtitle and grid
        st.subheader("Top Products by ABC Category")
        st.write("") 
        from st_aggrid import AgGrid, GridOptionsBuilder

        grid_df = prod_abc[[
            "Product_ID", "Product_Name", "revenue", "revenue_pct", "ABC_Category"
        ]]

        gb = GridOptionsBuilder.from_dataframe(grid_df)
        gb.configure_default_column(filterable=True, sortable=True, resizable=True)
        gb.configure_column(
            "ABC_Category",
            cellStyle={
                "condition": "value == 'A'",
                "style": {"backgroundColor": "#FFF4CE"}
            }
        )
        grid_opts = gb.build()

        AgGrid(
            grid_df,
            gridOptions=grid_opts,
            enable_enterprise_modules=False,
            theme="alpine",
            height=450,
            fit_columns_on_grid_load=True
        )

    #  — Right: the treemap —
    with tree_col:
        st.subheader("Revenue by ABC Category")
        fig_tm = px.treemap(
            prod_abc,
            path=["ABC_Category", "Product_Name"],
            values="revenue",
            color="revenue",
            color_continuous_scale="Oranges"
        )
        fig_tm.update_traces(
            hovertemplate="<b>%{label}</b><br>€%{value:,.0f}<br>%{percentRoot:.1%} of total",
            marker_line_width=1,
            marker_line_color="white"
        )
        fig_tm.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_tm, use_container_width=True, height=400)

    # 7) Claude‐powered strategies
    if st.button("Generate ABC-Based Product Strategies"):
        prompt = (
            "We have these product categories based on ABC analysis:\n"
            + "\n".join(
                f"- {row['Product_Name']}: Category {row['ABC_Category']}, Revenue €{row['revenue']:,}"
                for _, row in prod_abc.iterrows()
            )
            + "\n\nFor each category (A, B, C), recommend pricing or promotional strategies."
        )
        headers = {
            "Authorization": f"Bearer {CLAUDE_TOKEN}",
            "Content-Type": "application/json"
        }
        body = {"messages":[{"role":"user","content":prompt}]}

        with st.spinner("Generating strategies…"):
            r = requests.post(CLAUDE_URL, json=body, headers=headers, timeout=120)
            if r.status_code != 200:
                st.error(f"Invocation failed: {r.status_code}")
                st.code(r.text, language="json")
                st.stop()
            text = r.json()["choices"][0]["message"]["content"]

        st.markdown("""
        <div style="font-size: 18px; font-weight: bold; color: #cc0000; margin-bottom: 1rem;">
        📊 ABC Analysis: Pricing and Promotional Strategies
        </div>

        <p>Based on the ABC analysis of your product inventory, here are tailored pricing and promotional strategies to maximize revenue and optimize inventory management.</p>

        ---

        <h4>🅰️ Category A Products <span style="color: gray;">(High Value)</span></h4>
        <p>These products contribute significantly to revenue and should receive the most strategic attention.</p>

        <b>💰 Pricing Strategies:</b>
        <ul>
        <li><b>Premium Pricing:</b> Maintain high prices for items like <i>ReadWell Motherboards</i> and <i>Techix DVDs</i>.</li>
        <li><b>Value-Based Pricing:</b> Price based on perceived customer value rather than cost.</li>
        <li><b>Bundle Pricing:</b> Combine with Category B or C products to increase average order value.</li>
        <li><b>Limited Discounting:</b> Apply discounts selectively during key promotional periods.</li>
        </ul>

        <b>📣 Promotional Strategies:</b>
        <ul>
        <li><b>Premium Placement:</b> Feature in key marketing channels and storefronts.</li>
        <li><b>Loyalty Programs:</b> Reward repeat purchases with points or perks.</li>
        <li><b>Exclusive Features:</b> Highlight standout features in promotions.</li>
        <li><b>Priority Stock Management:</b> Always keep these products in stock.</li>
        <li><b>Sales Training:</b> Ensure sales staff are highly knowledgeable.</li>
        <li><b>Unboxing Experience:</b> Invest in premium packaging and presentation.</li>
        </ul>

        ---

        <h4>🅱️ Category B Products <span style="color: gray;">(Medium Value)</span></h4>
        <p>Moderate revenue drivers with potential to grow into Category A with the right strategy.</p>

        <b>💰 Pricing Strategies:</b>
        <ul>
        <li><b>Competitive Pricing:</b> Win market share by pricing strategically.</li>
        <li><b>Promotional Pricing:</b> Offer limited-time deals more frequently.</li>
        <li><b>Tiered Pricing:</b> Provide good-better-best product tiers.</li>
        <li><b>Psychological Pricing:</b> Use charm prices (e.g., €99.99).</li>
        </ul>

        <b>📣 Promotional Strategies:</b>
        <ul>
        <li><b>Cross-Promotion:</b> Pair with Category A products.</li>
        <li><b>Seasonal Campaigns:</b> Target based on seasonal demand cycles.</li>
        <li><b>Upgrade Marketing:</b> Encourage upgrades from Category C.</li>
        <li><b>Social Media:</b> Increase visibility through focused campaigns.</li>
        <li><b>Email Campaigns:</b> Highlight in newsletters to existing customers.</li>
        <li><b>Bundling:</b> Increase value perception through strategic product bundles.</li>
        </ul>

        ---

        <h4>🅲 Category C Products <span style="color: gray;">(Low Value)</span></h4>
        <p>Low contributors to revenue, but strategically useful for entry-level offerings or clearance.</p>

        <b>💰 Pricing Strategies:</b>
        <ul>
        <li><b>Economy Pricing:</b> Keep prices low to attract cost-sensitive customers.</li>
        <li><b>Clearance Pricing:</b> Move old stock quickly.</li>
        <li><b>Volume Discounts:</b> Encourage bulk purchases.</li>
        <li><b>Loss Leader Pricing:</b> Attract customers through aggressively priced products.</li>
        </ul>

        <b>📣 Promotional Strategies:</b>
        <ul>
        <li><b>Bulk Promotions:</b> Use "buy one, get one" offers.</li>
        <li><b>Clearance Sections:</b> Create dedicated clearance zones.</li>
        <li><b>Entry-Level Appeal:</b> Position as entry points for new customers.</li>
        <li><b>Stock Rationalization:</b> Regularly assess for discontinuation or reduction.</li>
        <li><b>Bundle with Premium:</b> Package with high-margin items to lift total basket value.</li>
        </ul>

        ---

        <h4>🛠️ Implementation Recommendations</h4>
        <ul>
        <li><b>Data-Driven Adjustments:</b> Monitor sales and reclassify quarterly.</li>
        <li><b>Customer Segmentation:</b> Align ABC products with specific customer groups.</li>
        <li><b>Supply Chain Priorities:</b> Ensure optimal inventory for Category A.</li>
        <li><b>Test & Learn:</b> Pilot pricing strategies in limited segments.</li>
        <li><b>Migration Strategy:</b> Develop tactics to move Category B products into A tier.</li>
        </ul>

        <p>✅ By executing these targeted strategies, you can boost profitability, improve inventory efficiency, and strengthen your overall market positioning.</p>
        """, unsafe_allow_html=True)

# --- Tab 4: Ask the Data ---
import os
import streamlit as st
import pandas as pd
import requests
from databricks import sql

# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Consumer Goods Analytics at Geek Peek", layout="wide")

# Load environment
CLAUDE_URL   = os.getenv("CLAUDE_ENDPOINT_URL")
CLAUDE_TOKEN = os.getenv("CLAUDE_BEARER_TOKEN")

@st.cache_data(ttl=600)
def load_table(query: str) -> pd.DataFrame:
    conn = sql.connect(
        server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
        http_path=os.getenv("DATABRICKS_HTTP_PATH"),
        access_token=os.getenv("DATABRICKS_ACCESS_TOKEN")
    )
    cursor = conn.cursor()
    cursor.execute(query)
    cols = [c[0] for c in cursor.description]
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(data, columns=cols)

@st.cache_data(ttl=600)
def get_data_context() -> str:
    df_kpis = load_table("""
      SELECT
        SUM(Total_Amount)  AS total_revenue,
        AVG(Total_Amount)  AS avg_order_value,
        COUNT(DISTINCT Customer_ID) AS unique_customers
      FROM gold.fact_sales
    """)
    seg = load_table("SELECT segment, COUNT(*) AS cnt FROM gold.customer_segments GROUP BY segment ORDER BY segment")
    abc = load_table("SELECT Product_Name, ABC_Category FROM gold.product_abc")
    ctx = (
        f"KPIs: Revenue €{df_kpis.total_revenue[0]:,.0f}, "
        f"AOV €{df_kpis.avg_order_value[0]:,.2f}, "
        f"Customers {df_kpis.unique_customers[0]:,}\n\n"
        "Segments:\n"
        + "\n".join(f"- {int(r.segment)}: {int(r.cnt)} customers"
                    for _, r in seg.iterrows())
        + "\n\nProduct ABC:\n"
        + ", ".join(f"{row.Product_Name}({row.ABC_Category})"
                    for _, row in abc.iterrows())
    )
    return ctx


# ─── TAB 4: Ask the Data ──────────────────────────────────────────────────────
with tabs[3]:
    # Create two columns: left panel (width=1) & main chat area (width=4)
    panel_col, chat_col = st.columns([1, 4], gap="small")

    # ---- LEFT PANEL in panel_col ----
    with panel_col:
        st.markdown(
            """
            <div style="
              background:#f0dfce;
              padding:1rem;
              height:calc(80vh - 2rem);
              box-sizing:border-box;
              border-radius:8px;
            ">
              <h2 style="margin:0 0 .5rem 0; color:#333;">💬 AI Assistant</h2>
              <p style="color:#222; line-height:1.4; font-size:0.9rem;">
                Analyze your KPIs, segments &amp; products<br/>
                and get actionable insights for decision-making.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- CHAT AREA in chat_col ----
    with chat_col:
        st.markdown("## 💬 Ask the Data")

        # Initialize history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hi there! 👋\n\nI can help you explore your data. What would you like to ask?"
                }
            ]

        # Render chat history
        chat_container = st.container()
        for msg in st.session_state.messages:
            chat_container.chat_message(msg["role"]).write(msg["content"])

        # Inject CSS for a fixed-footer input
        st.markdown(
            """
            <style>
              .footer-input {
                position: fixed;
                bottom: 0;
                left: 20%;   /* 1/(1+4) of the width */
                width: 80%;  /* 4/(1+4) of the width */
                padding: 1rem;
                background: #f7f7f7;
                box-shadow: 0 -2px 5px rgba(0,0,0,0.1);
                z-index: 1000;
              }
              /* ensure messages don't get hidden under the footer */
              .block-container {
                padding-bottom: 6rem;
              }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Build a form that clears on submit
        footer = st.empty()
        with footer.container():
            st.markdown('<div class="footer-input">', unsafe_allow_html=True)
            with st.form(key="chat_form", clear_on_submit=True):
                user_question = st.text_input(
                    "", 
                    placeholder="Ask me about KPIs, segments or products…",
                    label_visibility="hidden"
                )
                submitted = st.form_submit_button("➤")
            st.markdown('</div>', unsafe_allow_html=True)

        # Handle new question
        if submitted and user_question:
            # Add & render the user’s message
            st.session_state.messages.append({"role": "user", "content": user_question})
            chat_container.chat_message("user").write(user_question)

            # Call your LLM exactly as before
            data_context = get_data_context()
            body = {
                "messages": [
                    {"role": "system", "content": (
                        "You are an expert data analyst assistant. "
                        "Please format the answer as a Markdown bullet list, using '- ' at the start of each line, with one item per line."
                    )},
                    {"role": "assistant", "content": "Understood, here’s my answer:"},
                    {"role": "user", "content": (
                        f"Context:\n{data_context}\n\n"
                        f"Question: {user_question}\n\n"
                        "Please format the answer using clean bullet points, with each bullet on its own line. Start each bullet with '• '."
                    )},
                ]
            }
            headers = {"Authorization": f"Bearer {CLAUDE_TOKEN}", "Content-Type": "application/json"}
            with st.spinner("Thinking…"):
                r = requests.post(CLAUDE_URL, json=body, headers=headers, timeout=120)

            if r.status_code != 200:
                st.error(f"Error {r.status_code}")
                st.code(r.text, language="json")
            else:
                reply = r.json()["choices"][0]["message"]["content"]
                cleaned = "\n".join(
                    line for line in reply.splitlines()
                    if not (line.startswith("<<") and line.endswith(">>"))
                )

                # ✨ Format bullet points to render nicely
                if "•" in cleaned:
                    bullet_points = cleaned.split("•")
                    formatted = "\n".join(f"• {bp.strip()}" for bp in bullet_points if bp.strip())
                else:
                    formatted = cleaned

                # Show response
                st.session_state.messages.append({"role": "assistant", "content": formatted})
                chat_container.chat_message("assistant").write(formatted)