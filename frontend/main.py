import requests
import streamlit as st
import datetime
import json

def map_drought_category(prediction):
    categories = {
        0: '0 -> None, Normal or wet conditions',
        1: "1 -> D0, Abnormally Dry",
        2: "2 -> D1, Moderate Drought",
        3: "3 -> D2, Severe Drought",
        4: "4 -> D3, Extreme Drought", 
        5: "5 -> D4, Exceptional Drought",
    }
    return categories.get(prediction, "Unknown Category")

def main():
    # Activer le mode large
    st.set_page_config(layout="wide")

    st.title("Drought prediction from user input")
    st.subheader("Enter details below: ")

    # Formulaire avec deux colonnes
    with st.form("form", clear_on_submit=True):
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

            # Champs dans la première colonne
            with col1:
                fips = st.number_input("Enter FIPS code", value=1001, step=1)
                date = st.date_input("Enter date", datetime.date(2000, 1, 4))
                PRECTOT = st.number_input("Enter PRECTOT value", value=15.95)
                PS = st.number_input("Enter PS value", value=100.29)
                QV2M = st.number_input("Enter QV2M value", value=6.42)

            # Champs dans la deuxième colonne
            with col2:
                T2M = st.number_input("Enter T2M value", value=11.4)
                T2MDEW = st.number_input("Enter T2MDEW value", value=6.09)
                T2MWET = st.number_input("Enter T2MWET value", value=6.1)
                T2M_MAX = st.number_input("Enter T2M_MAX value", value=18.09)
                T2M_MIN = st.number_input("Enter T2M_MIN value", value=2.16)

            with col3:
                T2M_RANGE = st.number_input("Enter T2M_RANGE value", value=15.92)
                TS = st.number_input("Enter TS value", value=11.31)
                WS10M = st.number_input("Enter WS10M value", value=3.84)
                WS10M_MAX = st.number_input("Enter WS10M_MAX value", value=5.67)
                WS10M_MIN = st.number_input("Enter WS10M_MIN value", value=2.08)

            with col4:
                WS10M_RANGE = st.number_input("Enter WS10M_RANGE value", value=3.59)
                WS50M = st.number_input("Enter WS50M value", value=6.73)
                WS50M_MAX = st.number_input("Enter WS50M_MAX value", value=9.31)
                WS50M_MIN = st.number_input("Enter WS50M_MIN value", value=3.74)
                WS50M_RANGE = st.number_input("Enter WS50M_RANGE value", value=5.58)

            # Create the dictionary
            dd = {
                "fips": fips,
                "date": date.strftime("%Y-%m-%d"),  # Use isoformat for the date field
                "PRECTOT": PRECTOT,
                "PS": PS,
                "QV2M": QV2M,
                "T2M": T2M,
                "T2MDEW": T2MDEW,
                "T2MWET": T2MWET,
                "T2M_MAX": T2M_MAX,
                "T2M_MIN": T2M_MIN,
                "T2M_RANGE": T2M_RANGE,
                "TS": TS,
                "WS10M": WS10M,
                "WS10M_MAX": WS10M_MAX,
                "WS10M_MIN": WS10M_MIN,
                "WS10M_RANGE": WS10M_RANGE,
                "WS50M": WS50M,
                "WS50M_MAX": WS50M_MAX,
                "WS50M_MIN": WS50M_MIN,
                "WS50M_RANGE": WS50M_RANGE,
            }
            dd['date'] = datetime.datetime.combine(date, datetime.time.min).isoformat()
            


            # Submit button
            submit = st.form_submit_button("Predict")
            if submit:
                try:
                    # Make the POST request to the API
                    res = requests.post("http://127.0.0.1:8000/predict", data=json.dumps(dd), headers={"Content-Type": "application/json"})
                    # Check if the request was successful
                    if res.status_code == 200:
                        predictions = res.json().get("predictions")
                        st.write(f"##### {map_drought_category(predictions[0])}")
                    else:
                        st.error(f"Error: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
