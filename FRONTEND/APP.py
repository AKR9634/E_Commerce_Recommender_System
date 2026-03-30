import streamlit as st
import requests

API_URL = "http://api:8000"

st.title("🛒 E-Commerce Recommender System!!!")

user_type = st.radio("Are you a new user or existing user?", ["New User", "Existing User"])

user_id = "UNKNOWN_USER_ID"

if user_type == "Existing User":
    user_id = st.text_input("Enter your User ID:")

product_choice = st.radio("Do you have a product in mind?", ["Yes", "No"])

product_name = None
if product_choice == "Yes":
    product_name = st.text_input("Enter product name: ")

if st.button("Get Recommendations"):

    if user_type == "Existing User" and not user_id:
        st.warning("Please enter your User ID...")

    else:
        params = {}

        params["user"] = user_id

        if product_name:
            params["product"] = product_name

        try:
            response = requests.get(f"{API_URL}/recommend", params = params)
            response.raise_for_status()
            data = response.json()

            # st.write(data)
            for i, item in enumerate(data, 1):
                st.write(f"{i}. {item}")

        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")