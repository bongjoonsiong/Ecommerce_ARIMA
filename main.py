import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
import warnings
import io # Needed for reading uploaded file buffer

# Suppress convergence warnings for cleaner output (optional)
warnings.filterwarnings("ignore") 

# --- Streamlit App Configuration ---
st.set_page_config(page_title="ARIMA Sales Forecaster", layout="wide")
st.title("📈 ARIMA Time Series Sales Forecaster")

# --- Caching Functions ---
# Cache data loading and preprocessing to avoid re-running on every interaction
@st.cache_data
def load_data(uploaded_file):
    """Loads data from the uploaded CSV file."""
    try:
        # Use io.BytesIO for uploaded file compatibility
        bytes_data = uploaded_file.getvalue()
        df = pd.read_csv(io.BytesIO(bytes_data), encoding='ISO-8859-1')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocesses the loaded dataframe."""
    required_cols = ['Invoice', 'InvoiceDate', 'Quantity', 'Price']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        st.error(f"Error: Missing required columns in CSV: {missing}. Please ensure the file has the correct headers.")
        return None
   
    try:
        # Convert InvoiceDate to datetime objects
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        df.dropna(subset=['InvoiceDate'], inplace=True)

        # Remove cancellations/returns
        df_processed = df[~df['Invoice'].astype(str).str.startswith('C')].copy()
        df_processed = df_processed[df_processed['Quantity'] > 0]
        df_processed = df_processed[df_processed['Price'] > 0]

        # Calculate Total Sale
        df_processed['TotalSale'] = df_processed['Quantity'] * df_processed['Price']

        # Aggregate sales by day
        df_processed.set_index('InvoiceDate', inplace=True)
        daily_sales = df_processed['TotalSale'].resample('D').sum()
        daily_sales = daily_sales.fillna(0) # Fill missing days

        return daily_sales
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Cache ADF test results
@st.cache_data
def perform_adf_test(series):
    """Performs ADF test and returns results."""
    result = adfuller(series.dropna())
    return result

# Cache model fitting (can take time)
# Note: Caching models can sometimes be tricky. If issues arise, remove caching here.
@st.cache_data(show_spinner=False) # Show spinner manually later
def train_arima(series, order):
    """Trains the ARIMA model."""
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        st.error(f"Error fitting ARIMA model with order {order}: {e}")
        st.warning("Try adjusting the ARIMA order (p, d, q) in the sidebar.")
        return None

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file (e.g., Online Retail)", type=["csv"])

# Default values - adjustable by user
default_p = 5
default_q = 5
default_test_days = 30
default_forecast_days = 14

# --- Main Application Logic ---
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    df_raw = load_data(uploaded_file)

    if df_raw is not None:
        st.subheader("1. Raw Data Preview")
        st.dataframe(df_raw.head())

        st.subheader("2. Preprocessing & Daily Aggregation")
        with st.spinner("Preprocessing data and aggregating daily sales..."):
            daily_sales = preprocess_data(df_raw.copy()) # Use copy to avoid modifying cached raw data

        if daily_sales is not None and not daily_sales.empty:
            st.write(f"Time series generated from **{daily_sales.index.min().strftime('%Y-%m-%d')}** to **{daily_sales.index.max().strftime('%Y-%m-%d')}**.")
            st.dataframe(daily_sales.head())

            # --- 3. Time Series Visualization ---
            st.subheader("3. Visualizing Daily Sales")
            fig_raw, ax_raw = plt.subplots(figsize=(15, 6))
            ax_raw.plot(daily_sales.index, daily_sales, label='Total Daily Sales')
            ax_raw.set_title('Total Daily Sales Revenue Over Time')
            ax_raw.set_xlabel('Date')
            ax_raw.set_ylabel('Total Sales Revenue')
            ax_raw.legend()
            ax_raw.grid(True)
            st.pyplot(fig_raw)

            # --- 4. Stationarity Check ---
            st.subheader("4. Stationarity Check (ADF Test)")
            with st.spinner("Performing Augmented Dickey-Fuller test..."):
                adf_result = perform_adf_test(daily_sales)
                p_value = adf_result[1]

                st.write(f"ADF Statistic: `{adf_result[0]:.4f}`")
                st.write(f"P-value: `{p_value:.4f}`")
                st.write("Critical Values:")
                st.code(adf_result[4])

                d = 0 # Default order of differencing
                daily_sales_stationary = daily_sales # Assume stationary initially

                if p_value > 0.05:
                    st.warning("Series is likely non-stationary. Applying first-order differencing (d=1).")
                    d = 1
                    daily_sales_stationary = daily_sales.diff().dropna()
                    # Re-test after differencing
                    adf_result_diff = perform_adf_test(daily_sales_stationary)
                    st.write("--- ADF Test after Differencing (d=1) ---")
                    st.write(f"ADF Statistic: `{adf_result_diff[0]:.4f}`")
                    st.write(f"P-value: `{adf_result_diff[1]:.4f}`")
                    if adf_result_diff[1] <= 0.05:
                         st.success("Series is likely stationary after differencing.")
                    else:
                         st.error("Series still appears non-stationary after first differencing. Model may perform poorly. Consider further analysis or transformations.")
                else:
                    st.success("Series is likely stationary.")

            # --- 5. ACF/PACF Plots & Order Selection ---
            st.subheader("5. ACF/PACF Plots (for Order Estimation)")
            st.write(f"Plots based on data with differencing order `d={d}`.")
            fig_acf_pacf, axes = plt.subplots(1, 2, figsize=(12, 4))
            plot_acf(daily_sales_stationary, ax=axes[0], lags=40, title='Autocorrelation Function (ACF)')
            plot_pacf(daily_sales_stationary, ax=axes[1], lags=40, method='ywm', title='Partial Autocorrelation Function (PACF)')
            plt.tight_layout()
            st.pyplot(fig_acf_pacf)

            st.sidebar.subheader("ARIMA Order (p, d, q)")
            st.sidebar.info(f"Differencing order `d` determined as **{d}** from ADF test.")
            p = st.sidebar.number_input("Select AR order (p):", min_value=0, max_value=15, value=default_p, step=1, help="Suggested by PACF plot.")
            q = st.sidebar.number_input("Select MA order (q):", min_value=0, max_value=15, value=default_q, step=1, help="Suggested by ACF plot.")
            arima_order = (p, d, q)
            st.sidebar.write(f"Selected Order: **ARIMA{arima_order}**")

            st.sidebar.subheader("Forecasting Parameters")
            test_days = st.sidebar.number_input("Days for Test Set:", min_value=7, max_value=365, value=default_test_days, step=1)
            forecast_days = st.sidebar.number_input("Days to Forecast:", min_value=1, max_value=90, value=default_forecast_days, step=1)

            # --- 6. Model Training & Prediction ---
            st.subheader(f"6. Training ARIMA{arima_order} Model")
            if len(daily_sales) > test_days:
                train_data = daily_sales[:-test_days]
                test_data = daily_sales[-test_days:]

                # Ensure frequency is set for statsmodels
                if train_data.index.freq is None:
                    train_data = train_data.asfreq('D')

                with st.spinner(f"Fitting ARIMA{arima_order} model to {len(train_data)} data points... This may take a moment."):
                    model_fit = train_arima(train_data, arima_order)

                if model_fit:
                    st.success("Model training complete.")
                    st.text_area("Model Summary", model_fit.summary().as_text(), height=300)

                    # --- 7. Prediction & Evaluation ---
                    st.subheader("7. Prediction on Test Set")
                    with st.spinner("Generating predictions on the test set..."):
                        try:
                            start_index = len(train_data)
                            end_index = len(train_data) + len(test_data) - 1
                            predictions = model_fit.predict(start=start_index, end=end_index)
                            if isinstance(test_data.index, pd.DatetimeIndex):
                                predictions.index = test_data.index # Align index

                            rmse = np.sqrt(mean_squared_error(test_data, predictions))
                            st.metric(label="Test Set RMSE", value=f"{rmse:.2f}")
                            st.write("Lower RMSE indicates better fit to the test data.")

                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                            predictions = None

                    # --- 8. Forecasting Future Values ---
                    st.subheader(f"8. Forecasting Next {forecast_days} Days")
                    with st.spinner(f"Refitting model on full data and forecasting {forecast_days} days..."):
                        try:
                            # Refit on full data
                            if daily_sales.index.freq is None:
                                full_daily_sales = daily_sales.asfreq('D')
                            else:
                                full_daily_sales = daily_sales

                            full_model_fit = train_arima(full_daily_sales, arima_order) # Use cached function if possible

                            if full_model_fit:
                                # Generate forecast
                                forecast = full_model_fit.predict(start=len(full_daily_sales), end=len(full_daily_sales) + forecast_days - 1)
                                last_date = full_daily_sales.index[-1]
                                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
                                forecast.index = forecast_index
                                st.dataframe(forecast.to_frame(name='Forecasted Sales'))
                            else:
                                st.warning("Could not refit model on full data for forecasting.")
                                forecast = None

                        except Exception as e:
                            st.error(f"Error during forecasting: {e}")
                            forecast = None


                    # --- 9. Final Visualization ---
                    st.subheader("9. Visualization: History, Predictions & Forecast")
                    fig_final, ax_final = plt.subplots(figsize=(15, 7))
                    ax_final.plot(daily_sales.index, daily_sales, label='Historical Daily Sales', color='blue')
                    if predictions is not None:
                        ax_final.plot(predictions.index, predictions, label=f'ARIMA Predictions (Test Set)', color='orange')
                    if forecast is not None:
                        ax_final.plot(forecast.index, forecast, label=f'ARIMA Forecast', color='red', linestyle='--')

                    ax_final.set_title('Daily Sales: Historical, Predictions, and Forecast')
                    ax_final.set_xlabel('Date')
                    ax_final.set_ylabel('Total Sales Revenue')
                    ax_final.legend()
                    ax_final.grid(True)
                    st.pyplot(fig_final)

                else:
                    st.error("Model training failed. Cannot proceed with prediction and forecasting.")
            else:
                st.warning(f"Not enough data ({len(daily_sales)} days) for the specified test period ({test_days} days). Please reduce the test period or upload a larger dataset.")
        else:
            st.warning("Could not generate daily sales time series from the preprocessed data.")
    else:
        st.info("Awaiting data file upload...")

else:
    st.info("Please upload a CSV file using the sidebar to begin forecasting.")

st.sidebar.markdown("---")
st.sidebar.markdown("Created based on ARIMA time series analysis.")

