import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from create_time_features import create_time_features
from create_time_features import create_lag_features
from create_time_features import X_train

# -- Configuration --
MODEL = 'demand_forecast_model.pkl'
DATA = pd.read_csv('full_training_data.csv', parse_dates=['date'])
UNIQUE_ITEMS = sorted(DATA['item'].unique())
UNIQUE_STORES = sorted(DATA['store'].unique())

# -- Feature Generation Functions (must match training) --
def create_future_features(start_date, forecast_days, store, item, historical_data):
    future_dates = pd.date_range(start=start_date, periods=forecast_days)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['store'] = store
    future_df['item'] = item
    future_df = create_time_features(future_df)
    
    # Append future_df to historical data for lag features
    combined_df = pd.concat([historical_data, future_df], ignore_index=True)
    combined_df = create_lag_features(combined_df, lags=[7, 28])

    # Filter back to only future dates
    future_df_final = combined_df[combined_df['date'] >= start_date].reset_index(drop=True).copy()

    # Keep a safe copy of the date column
    date_col = future_df_final['date']

    # Select only necessary features to match model training
    X_future = future_df_final[X_train.columns]

    return X_future, date_col


# -- Streamlit App --

st.title("Inventory Demand Forecasting Web App")
st.markdown("Forecast future demand for items in stores using a pre-trained LightGBM model.")

# -- User Inputs --
with st.sidebar:
    st.header("Forecast Parameters")
    selected_store = st.selectbox("Select Store", UNIQUE_STORES)
    selected_item = st.selectbox("Select Item", UNIQUE_ITEMS)
    forecast_days = st.slider("Select Forecast Horizon (days)", min_value=7, max_value=60, value=30, step=1)

if st.button("Predict Demand"):
    # 1. Get the last date in the historical data for the selected store and item
    last_date = DATA[(DATA['store'] == selected_store) & (DATA['item'] == selected_item)]['date'].max()
    start_forecast_date = last_date + timedelta(days=1)

    with st.spinner("Generating forecast..."):
        # 2. Create future features and extract date
        X_future, future_dates = create_future_features(start_forecast_date, forecast_days, selected_store, selected_item, DATA)

        # 3. Load the pre-trained model
        model = joblib.load(MODEL)

        # 4. Make predictions
        predictions = model.predict(X_future)

        # 5. Create results DataFrame
        forecast_results = pd.DataFrame({
            'date': future_dates.iloc[:forecast_days],
            'predicted_demand': np.round(predictions).astype(int)
        })

        # 6. Get historical data for plotting
        historical_data = DATA[(DATA['store'] == selected_store) & (DATA['item'] == selected_item)].tail(90).copy()
        historical_data = historical_data[['date', 'demand']].rename(columns={'demand': 'predicted_demand'})

        # 7. Combine historical and forecast data for visualization
        plot_data = pd.concat([historical_data, forecast_results], ignore_index=True)
        plot_data['type'] = ['Historical'] * len(historical_data) + ['Forecast'] * len(forecast_results)

    # 8. Display results
    st.subheader(f"Forecasted Demand for {selected_item} in Store {selected_store}")

    # 9. Display forecast table
    st.dataframe(forecast_results.set_index('date'))

    # 10. Display chart
    import altair as alt

    chart = alt.Chart(plot_data).encode(
        x=alt.X('date:T'),
        y=alt.Y('predicted_demand:Q'),
        color=alt.Color('type:N', scale=alt.Scale(domain=['Historical', 'Forecast'], range=['blue', 'orange']))
    )

    line = chart.mark_line().properties(
        title=f"Historical and Forecasted Demand for {selected_item} in Store {selected_store}"
    )

    # Adding a vertical line for forecast boundary
    forecast_boundary = alt.Chart(pd.DataFrame({
        'date': [start_forecast_date]
    })).mark_rule(color='red', strokeDash=[5,5]).encode(
        x='date:T'
    )

    st.altair_chart(line + forecast_boundary, use_container_width=True)
