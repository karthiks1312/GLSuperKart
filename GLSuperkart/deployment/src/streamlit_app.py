import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ---- Streamlit bootstrap ----
st.set_page_config(page_title="GLSuperKart Sales Prediction")

# ---- Read secrets ----
REPO_ID = os.getenv("HF_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

if not REPO_ID:
    st.error("REPO_ID secret is missing in HF Space")
    st.stop()

# ---- Render UI immediately ----
st.title("🛒 SuperKart Sales Prediction")
st.write("UI rendered successfully")

# ---- Load model lazily ----

repo_path=f"{REPO_ID}/superkart-sales-model"

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=repo_path,
            filename="best_superkart_sales_model_v1.joblib",
            repo_type="model",
            token=HF_TOKEN
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error("Failed to download model from Hugging Face")
        st.exception(e)
        st.stop()

model = load_model()


# ---- UI ----
st.write("""
This application predicts the **total product sales** for SuperKart
based on product characteristics and store attributes.
""")

product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_type = st.selectbox("Product Type", [
    "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy",
    "Frozen Foods", "Fruits and Vegetables", "Hard Drinks",
    "Health and Hygiene", "Household", "Meat", "Others", "Seafood",
    "Snack Foods", "Soft Drinks", "Starchy Foods"
])
store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
store_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])

product_weight = st.number_input("Product Weight (kg)", 1.0, 50.0, 12.0)
product_allocated_area = st.number_input("Product Allocated Area", 0.001, 1.0, 0.05, step=0.001)
product_mrp = st.number_input("Product MRP", 1.0, 1000.0, 100.0)
store_est_year = st.number_input("Store Establishment Year", 1950, 2025, 2000)

input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Allocated_Area": product_allocated_area,
    "Product_MRP": product_mrp,
    "Store_Establishment_Year": store_est_year,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Type": product_type,
    "Store_Id": store_id,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_city_type,
    "Store_Type": store_type
}])

if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Product Sales: **₹ {prediction:,.2f}**")
