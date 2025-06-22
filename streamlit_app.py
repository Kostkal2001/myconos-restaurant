import pandas as pd
import streamlit as st
import os
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ============ SETTINGS =============
SUPPORT_THRESHOLD = 0.001
LIFT_THRESHOLD = 0.5
# ===================================

st.set_page_config(page_title="Meals by Genet - Recommender", layout="wide")
st.markdown("""
    <style>
    /* Γενικά */
    body {
        background-color: #111111;
        color: #f1f1f1;
    }

    .block-container {
        padding-top: 2rem;
        background-color: #111111;
    }

    h1, h2, h3, h4 {
        color: gold !important;
        font-family: 'Georgia', serif;
    }

    /* Εισαγωγές, επιλογές, sliders */
    .stTextInput label, .stSelectbox label, .stSlider label, .stTextArea label, .stNumberInput label {
        color: white !important;
    }

    .stMarkdown, .stTextInput, .stSelectbox, .stSlider, .stTextArea, .stNumberInput {
        font-size: 16px;
        color: #f5f5f5;
    }

    input, textarea, select {
        color: black !important;
        background-color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: gold;
        color: black;
        font-weight: bold;
        border: none;
    }

    .stButton > button:hover {
        background-color: #e6c200;
    }

    /* Power BI iframe */
    iframe {
        border: 2px solid gold;
        border-radius: 10px;
    }

    /* Tabs - Χρυσά όλα */
            div[data-testid="stTabs"] div[role="tablist"] {
    margin-top: 20px !important;
}
    div[data-testid="stTabs"] div[role="tablist"] > button {
    color: #ccc !important;
    background-color: transparent !important;
    border: none !important;
    padding: 10px 20px;
    margin-right: 8px;
    border-radius: 8px 8px 0 0;
    font-size: 16px;
    font-weight: 500;
    transition: all 0.3s ease;
}

div[data-testid="stTabs"] div[role="tablist"] > button:hover {
    color: gold !important;
    background-color: #222 !important;
}

div[data-testid="stTabs"] div[role="tablist"] > button[aria-selected="true"] {
    background-color: #222 !important;
    border-bottom: 3px solid gold !important;
    font-weight: 700 !important;
    color: gold !important;
    box-shadow: 0 -2px 10px rgba(255, 215, 0, 0.3);
}


    /* Τίτλοι section */
    .stApp h2 {
        color: gold !important;
    }

    /* Expander headers - always visible + luxury look */
    summary {
        color: white !important;
        font-size: 17px !important;
        font-weight: bold !important;
        background-color: #222 !important;
        padding: 8px;
        border-radius: 6px;
        border: 1px solid gold;
        list-style: none;
        cursor: pointer;
        box-shadow: 0 0 8px rgba(255, 215, 0, 0.15);
    }

    summary:hover {
        background-color: #333 !important;
        transition: background-color 0.3s ease;
    }

    details {
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)





@st.cache_data
def load_data():
    return pd.read_csv("SalesMBA_cleaned.csv")

def split_by_time(df):
    return {
        "Breakfast": df[(df['Receipt Time HH'] >= 6) & (df['Receipt Time HH'] <= 11)],
        "Lunch": df[(df['Receipt Time HH'] >= 12) & (df['Receipt Time HH'] <= 17)],
        "Dinner": df[(df['Receipt Time HH'] >= 18) & (df['Receipt Time HH'] <= 23)],
        "Late Night": df[(df['Receipt Time HH'] >= 0) & (df['Receipt Time HH'] <= 5)],
    }

@st.cache_data
def run_market_basket(data, min_support=SUPPORT_THRESHOLD, min_lift=LIFT_THRESHOLD):
    baskets = data.groupby("Order Number")["Product"].apply(list).values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules["rule_tuple"] = rules.apply(lambda x: frozenset([frozenset(x["antecedents"]), frozenset(x["consequents"])]), axis=1)
    rules = rules.drop_duplicates(subset="rule_tuple").drop(columns=["rule_tuple"])
    return rules.sort_values(by="lift", ascending=False)

df = load_data()
splits = split_by_time(df)

tab0, tab1, tab2, tab3, tab4 = st.tabs(["Home", "Recommendations", "Restaurant Menu", "Statistics", "Order Now"])


with tab0:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("Celebrities01.jpg", use_container_width=True)
    st.markdown("""
    <h1 style='text-align: center;'>Myconos Restaurant</h1>
    <p style='text-align: center; font-size: 18px;'>
        Καλώς ήρθατε στο εστιατόριό μας. Εδώ μπορείτε να δείτε προτάσεις για γεύματα με βάση τις προτιμήσεις των πελατών μας και την ώρα της ημέρας.<br>
        Εξερευνήστε το μενού μας, ανακαλύψτε τοπικά προϊόντα και ζήστε την εμπειρία Meals by Genet.
    </p>
    """, unsafe_allow_html=True)
    st.subheader("Contact")
    st.markdown("""
📞 Phone: +30 233445465757  
📍 Location: Myconos, Greece  
📧 Email: info@myconosrestaurant.gr  
🕒 Opening hours: 08:00 - 00:00 
""")
    st.markdown("---")
    st.markdown("Εάν έχετε οποιαδήποτε απορία ή θέλετε να κάνετε κράτηση, παρακαλούμε επικοινωνήστε μαζί μας ή συμπληρώστε τη φόρμα παρακάτω:")
    name = st.text_input("Όνομα")
    email = st.text_input("Email")
    message = st.text_area("Μήνυμα ή αίτημα για κράτηση")
    if st.button("Αποστολή"):
        st.success("Το μήνυμά σας εστάλη επιτυχώς! Θα επικοινωνήσουμε μαζί σας σύντομα.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab1:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.header("Προτεινόμενα προϊόντα")
    st.markdown("</div>", unsafe_allow_html=True)

    time_selected = st.selectbox("Επίλεξε χρονική ζώνη", list(splits.keys()))
    filtered_df = splits[time_selected]
    rules = run_market_basket(filtered_df)

    all_products = sorted(set(filtered_df["Product"].tolist()))
    search_input = st.text_input("Αναζήτησε προϊόν:").strip().upper()

    if search_input:
        import difflib
        contains_matches = [p for p in all_products if search_input in p]
        fuzzy_matches = difflib.get_close_matches(search_input, all_products, n=10, cutoff=0.4)
        combined_matches = sorted(set(contains_matches + fuzzy_matches))
    else:
        combined_matches = all_products

    selected_product = st.selectbox("Επίλεξε προϊόν από αποτελέσματα:", combined_matches)

    product_category = df[df['Product'] == selected_product]['Category'].values[0]
    local_alternatives = df[
        (df['Category'] == product_category) &
        (df['Is_Local'] == 'Yes') &
        (df['Product'] != selected_product)
    ]['Product'].unique()

    if len(local_alternatives) > 0:
        st.markdown("Θες να το ανταλλάξεις με ένα τοπικό προϊόν;")
        chosen_local = st.selectbox("Διαθέσιμες τοπικές επιλογές:", local_alternatives)
        if st.button("Αντάλλαξε με Local"):
            st.success(f"Αντί για {selected_product}, προτείνουμε: {chosen_local} (Local)")

    st.markdown("---")

    recommended = rules[rules['antecedents'].apply(lambda x: selected_product in x)]

    if not recommended.empty:
        st.subheader("Προτεινόμενα προϊόντα:")

        unique_recommendations = {}
        for _, row in recommended.iterrows():
            for item in row['consequents']:
                if item not in unique_recommendations:
                    unique_recommendations[item] = (row['confidence'], row['lift'])

        st.markdown("Φίλτρα για Confidence και Lift")
        col1, col2 = st.columns(2)
        min_conf = col1.slider("Ελάχιστο Confidence", 0.0, 1.0, 0.1, 0.01)
        min_lift = col2.slider("Ελάχιστο Lift", 0.0, 5.0, 1.0, 0.1)

        filtered_recs = [(p, (c, l)) for p, (c, l) in unique_recommendations.items() if c >= min_conf and l >= min_lift]

        for prod, (conf, lift) in sorted(filtered_recs, key=lambda x: -x[1][1]):
            tags = ""

            prod_info = filtered_df[filtered_df["Product"] == prod]
            if not prod_info.empty:
                category = prod_info["Category"].values[0]
                is_local = prod_info["Is_Local"].values[0]

                if "Alcohol" in category:
                    tags += "🍷 "
                elif "Appetizers" in category:
                    tags += "🧀 "
                elif "Breakfast" in category:
                    tags += "🍳 "
                elif "Coffee" in category:
                    tags += "☕ "
                elif "Desserts" in category:
                    tags += "🍰 "
                elif "Juices" in category:
                    tags += "🧃 "
                elif "Main Dishes" in category:
                    tags += "🍽️ "
                elif "Salads" in category:
                    tags += "🥗 "
                elif "Sushi" in category:
                    tags += "🍣 "

                if is_local == "Yes":
                    tags += "📍 "
                if "VEGAN" in category.upper() or "VEGETARIAN" in category.upper():
                    tags += "🌿 "
                if lift > 5:
                    tags += "🔥 "

            st.markdown(f"""
            <div style="text-align: center; font-size: 16px;">
                <h4 style="margin-bottom: 0;">{tags}{prod}</h4>
                <ul style="list-style-position: inside; padding-left: 0;">
                    <li><strong>Confidence:</strong> {conf:.2f} → Από 100 άτομα που πήραν το <strong>{selected_product}</strong>, περίπου {conf*100:.0f} πήραν και αυτό το προϊόν.</li>
                    <li><strong>Lift:</strong> {lift:.2f} → Το προϊόν εμφανίζεται {lift:.2f} φορές πιο συχνά μαζί με το επιλεγμένο, σε σχέση με το αν ήταν τυχαίο.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)

    else:
        st.info("Δεν βρέθηκαν σχετικές προτάσεις μέσω MBA. Εμφανίζουμε εναλλακτικές με βάση το περιεχόμενο (Content-Based).")

        prod_info = filtered_df[filtered_df["Product"] == selected_product]
        if not prod_info.empty:
            category = prod_info["Category"].values[0]
            group = prod_info["Group"].values[0]
            fallback_products = (
                filtered_df[
                    (filtered_df["Category"] == category) &
                    (filtered_df["Group"] == group) &
                    (filtered_df["Product"] != selected_product)
                ]["Product"]
                .value_counts()
                .head(10)
            )

            if not fallback_products.empty:
                st.subheader("Content-Based Προτάσεις:")
                for p, count in fallback_products.items():
                    tags = ""
                    info = filtered_df[filtered_df["Product"] == p]
                    if not info.empty:
                        cat = info["Category"].values[0]
                        loc = info["Is_Local"].values[0]

                        if "Alcohol" in cat:
                            tags += "🍷 "
                        elif "Appetizers" in cat:
                            tags += "🧀 "
                        elif "Breakfast" in cat:
                            tags += "🍳 "
                        elif "Coffee" in cat:
                            tags += "☕ "
                        elif "Desserts" in cat:
                            tags += "🍰 "
                        elif "Juices" in cat:
                            tags += "🧃 "
                        elif "Main Dishes" in cat:
                            tags += "🍽️ "
                        elif "Salads" in cat:
                            tags += "🥗 "
                        elif "Sushi" in cat:
                            tags += "🍣 "

                        if loc == "Yes":
                            tags += "📍 "
                        if "VEGAN" in cat.upper() or "VEGETARIAN" in cat.upper():
                            tags += "🌿 "

                    st.markdown(f"""
                    <div style="text-align: center; font-size: 16px;">
                        <h4 style="margin-bottom: 0;">{tags}{p}</h4>
                        <p>Παραγγέλθηκε <strong>{count} φορές</strong> από πελάτες που προτίμησαν παρόμοια προϊόντα με το <strong>{selected_product}</strong>.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("<hr>", unsafe_allow_html=True)
            else:
                st.warning("Δεν βρέθηκαν παρόμοια προϊόντα για fallback.")
        else:
            st.warning("Δεν βρέθηκαν πληροφορίες για το προϊόν.")



with tab2:
    st.header("Restaurant Menu by Category")

    # Αντιστοίχιση κατηγοριών με emoji
    category_emojis = {
        "Alcohol": "🍷",
        "Appetizers & Sides": "🧀",
        "Breakfast & Brunch": "🍳",
        "Coffee & Milkshakes": "☕",
        "Desserts": "🍰",
        "Juices & Soft Drinks": "🧃",
        "Main Dishes": "🍽️",
        "Salads": "🥗",
        "Sushi": "🍣"
    }

    # Ταξινομημένες κατηγορίες
    categories = sorted(df['Category'].dropna().unique())

    for cat in categories:
        emoji = category_emojis.get(cat, "🍽️")  # default emoji αν δεν βρεθεί
        with st.expander(f" {cat} {emoji}", expanded=False):
            items = sorted(df[df['Category'] == cat]['Product'].unique())
            for item in items:
                st.markdown(f"- {item}")


with tab3:
    st.header("📊 Restaurant Statistics - Power BI")

    st.markdown("""
    Here you can see useful insights about orders, product categories, time slot performance, and other key restaurant analytics generated from the system.    """, unsafe_allow_html=True)

    # ✅ Αντικατέστησε το παρακάτω link με το δικό σου από Power BI
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=8e1e7eee-0afb-47d5-82c4-6be79ae2eb3c&autoAuth=true&ctid=54a6385f-8ade-4892-a404-d486b55a6746"

    st.markdown(f"""
    <iframe title="Power BI Report" width="100%" height="700" src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>
    """, unsafe_allow_html=True)
from datetime import datetime

# Ανάλογα με την ώρα, βρες τη σωστή ζώνη
def get_time_slot():
    hour = datetime.now().hour
    if 6 <= hour <= 11:
        return "Breakfast"
    elif 12 <= hour <= 17:
        return "Lunch"
    elif 18 <= hour <= 23:
        return "Dinner"
    else:
        return "Late Night"

# Αυτό είναι το split που θα χρησιμοποιηθεί στο tab4
current_slot = get_time_slot()
filtered_df_order = splits[current_slot]
rules_order = run_market_basket(filtered_df_order)

with tab4:
    st.header("🛒 Place Your Order")

    # Φόρμα παραγγελίας
    with st.form("order_form"):
        customer_name = st.text_input("👤 Your Name")
        table_number = st.text_input("🍽️ Table Number")
        phone_number = st.text_input("📱 Mobile Number")

        all_products = sorted(df["Product"].dropna().unique())
        selected_products = st.multiselect("🧾 Select Products to Order", all_products)

        # MBA Προτάσεις (μέσα στο form για live preview)
        st.markdown("---")
        rules_order = run_market_basket(df)

        if selected_products:
            st.subheader("🤖 Suggested Add-ons Based on Your Selection")

            recommendations = {}
            for prod in selected_products:
                recs = rules_order[rules_order['antecedents'].apply(lambda x: prod in x)]
                for _, row in recs.iterrows():
                    for consequent in row['consequents']:
                        if consequent not in selected_products:
                            if consequent not in recommendations:
                                recommendations[consequent] = (row['confidence'], row['lift'])

            sorted_recs = sorted(recommendations.items(), key=lambda x: -x[1][1])

            if sorted_recs:
                for prod, (conf, lift) in sorted_recs[:5]:  # Top 5 προτάσεις
                    st.markdown(f"🔁 **{prod}** – Confidence: {conf:.2f}, Lift: {lift:.2f}")
            else:
                st.info("Δεν υπάρχουν σχετικές προτάσεις για τα προϊόντα που επιλέχθηκαν.")

        st.markdown("---")
        notes = st.text_area("📝 Additional Notes (optional)", "")
        submitted = st.form_submit_button("✅ Submit Order")

    # Υποβολή παραγγελίας
    if submitted:
        if not customer_name or not table_number or not phone_number or not selected_products:
            st.warning("⚠️ Please fill in all required fields and select at least one product.")
        else:
            st.success(f"🎉 Thank you, {customer_name}! Your order has been submitted.")
            st.markdown(f"""
            🧾 **Order Summary:**  
            • **Table:** {table_number}  
            • **Phone:** {phone_number}  
            • **Products:** {', '.join(selected_products)}  
            • **Notes:** {notes if notes else 'None'}  
            """)
