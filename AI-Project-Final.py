# AI-Project-Final.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def main():
    # Load preprocessed weekly average prices
    data_path = "weekly_avg_prices-2.csv"
    weekly_avg = pd.read_csv(data_path)
    weekly_avg['Week'] = pd.to_datetime(weekly_avg['Week'])

    # Get available crops
    crops = weekly_avg['Commodity'].str.lower().unique()
    print("\nAvailable Crops:")
    for i, crop in enumerate(crops, 1):
        print(f"{i}. {crop.capitalize()}")

    # --- User selects crop ---
    while True:
        crop_input = input("\nEnter crop name (e.g., tomato): ").strip().lower()
        if crop_input in crops:
            break
        print("Invalid crop name. Try again.")

    # Prepare time series data
    crop_df = weekly_avg[weekly_avg['Commodity'].str.lower() == crop_input]
    crop_df = crop_df.sort_values('Week')
    ts = crop_df.set_index('Week')['Weekly_Avg_Modal_Price']

    # Fit ARIMA model
    model = ARIMA(ts, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.get_forecast(steps=26)
    forecast_df = forecast.summary_frame()
    forecast_df['Week'] = pd.date_range(start=ts.index[-1] + pd.Timedelta(weeks=1), periods=26, freq='W')
    forecast_df.set_index('Week', inplace=True)

    # Plot historical and forecasted prices
    plt.figure(figsize=(10, 5))
    ts.plot(label='Observed')
    forecast_df['mean'].plot(label='Forecast', color='green')
    plt.title(f"Forecasted Prices for {crop_input.capitalize()} (Next 6 Months)")
    plt.xlabel("Week")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Allow user to query expected price for a specific date
    while True:
        date_input = input("\nEnter a future date (YYYY-MM-DD) to check expected price (or 'exit'): ").strip()
        if date_input.lower() == 'exit':
            break
        try:
            query_date = pd.to_datetime(date_input)
            # Align query_date to the nearest 'W-SUN' to match forecast_df index frequency
            query_date = query_date.normalize()  # Set time to midnight
            # Find closest forecast index
            closest = forecast_df.index[abs(forecast_df.index - query_date) < pd.Timedelta(days=4)]
            if not closest.empty:
                price = forecast_df.loc[closest[0], 'mean']
                print(f"\nExpected modal price for {crop_input.capitalize()} on {closest[0].date()} is ₹{price:.2f}")
            else:
                print("Date not found in forecast range. Please choose a date within next 6 months (weekly interval).")
        except Exception:
            print("Invalid date format. Please use YYYY-MM-DD.")

    print("\nThank you for using the crop price forecaster!")

if __name__ == "__main__":
    main()
