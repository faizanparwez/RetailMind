
import os
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "../data/train.csv"

MODEL_DIR = "../models"
OUTPUT_DIR = "../outputs/day9"

MODEL_PATH = os.path.join(MODEL_DIR, "demand_forecaster.keras")

LOOK_BACK = 30
TEST_DAYS = 28
TOP_N = 20

EPOCHS = 30
BATCH_SIZE = 64

tf.keras.utils.set_random_seed(42)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD AND INSPECT DATA
# ============================================================

print("\nLoading Store Item Demand dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = ["date", "store", "item", "sales"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

df["date"] = pd.to_datetime(df["date"])
df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

df = df.dropna(subset=["date", "store", "item", "sales"])

df["sales"] = df["sales"].clip(lower=0)

print("\nDataset shape:", df.shape)
print("\nDataset preview:")
print(df.head())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())


# ============================================================
# 2. SELECT TOP STORE-ITEM COMBINATIONS
# ============================================================

print("\nSelecting top-selling store-item combinations...")

top_combinations = (
    df.groupby(["store", "item"])["sales"]
    .sum()
    .nlargest(TOP_N)
    .index.tolist()
)

print("Selected combinations:", len(top_combinations))

all_dates = pd.date_range(
    start=df["date"].min(),
    end=df["date"].max(),
    freq="D"
)

cutoff_date = all_dates.max() - pd.Timedelta(days=TEST_DAYS)

print("\nTraining data ends:", cutoff_date)
print("Testing period starts:", cutoff_date + pd.Timedelta(days=1))


# ============================================================
# 3. CREATE DAILY TIME SERIES AND SUPERVISED WINDOWS
# ============================================================

print("\nPreparing time-series windows...")

train_x = []
train_y = []

test_x = []
test_info = []

series_metadata = []

for store, item in top_combinations:

    combo_df = df[
        (df["store"] == store) &
        (df["item"] == item)
    ]

    # Aggregate sales by day.
    daily_sales = (
        combo_df.groupby("date")["sales"]
        .sum()
        .reindex(all_dates, fill_value=0)
    )

    # Use training-period maximum for this series' scaling.
    training_sales = daily_sales.loc[:cutoff_date]

    scale = float(training_sales.max())

    if scale <= 0:
        scale = 1.0

    scaled_sales = (daily_sales / scale).tolist()

    series_metadata.append({
        "store": store,
        "item": item,
        "scale": scale
    })

    # Create look-back windows.
    for i in range(LOOK_BACK, len(scaled_sales)):

        window = scaled_sales[i - LOOK_BACK:i]
        target_date = all_dates[i]

        if target_date <= cutoff_date:

            train_x.append(window)
            train_y.append(scaled_sales[i])

        else:

            test_x.append(window)

            test_info.append({
                "store": store,
                "item": item,
                "date": target_date,
                "actual_scaled": scaled_sales[i],
                "scale": scale
            })


if len(train_x) == 0 or len(test_x) == 0:
    raise ValueError(
        "Not enough data to create training or testing windows."
    )

print("Training windows:", len(train_x))
print("Testing windows:", len(test_x))


# Convert data into TensorFlow tensors.
X_train = tf.convert_to_tensor(train_x, dtype=tf.float32)
y_train = tf.convert_to_tensor(train_y, dtype=tf.float32)

X_test = tf.convert_to_tensor(test_x, dtype=tf.float32)

# LSTM expects: (samples, timesteps, features)
X_train = tf.expand_dims(X_train, axis=-1)
X_test = tf.expand_dims(X_test, axis=-1)

print("\nTraining input shape:", X_train.shape)
print("Testing input shape:", X_test.shape)


# ============================================================
# 4. BUILD LSTM MODEL
# ============================================================

print("\nBuilding LSTM model...")

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(LOOK_BACK, 1)
    ),

    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(
        32
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1
    )

])

model.compile(
    optimizer="adam",
    loss="mae",
    metrics=["mae"]
)

model.summary()


# ============================================================
# 5. TRAIN THE MODEL
# ============================================================

print("\nTraining LSTM...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[early_stopping],
    shuffle=True,
    verbose=1
)

print("\nTraining completed!")
print("Epochs trained:", len(history.history["loss"]))


# ============================================================
# 6. PREDICT ON THE HELD-OUT TEST PERIOD
# ============================================================

print("\nGenerating test predictions...")

predictions_scaled = model.predict(
    X_test,
    verbose=0
).flatten()

actual_values = []
predicted_values = []
prediction_records = []

for i, info in enumerate(test_info):

    scale = info["scale"]

    actual = info["actual_scaled"] * scale
    predicted = max(0, float(predictions_scaled[i]) * scale)

    actual_values.append(actual)
    predicted_values.append(predicted)

    prediction_records.append({
        "date": info["date"],
        "store": info["store"],
        "item": info["item"],
        "actual_sales": actual,
        "predicted_sales": predicted
    })

predictions_df = pd.DataFrame(prediction_records)

predictions_df = predictions_df.sort_values(
    ["store", "item", "date"]
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

print("\nEvaluating model...")

actual_tensor = tf.convert_to_tensor(
    actual_values,
    dtype=tf.float32
)

predicted_tensor = tf.convert_to_tensor(
    predicted_values,
    dtype=tf.float32
)

mae = tf.reduce_mean(
    tf.abs(actual_tensor - predicted_tensor)
)

rmse = tf.sqrt(
    tf.reduce_mean(
        tf.square(actual_tensor - predicted_tensor)
    )
)

print("\nLSTM DEMAND FORECASTING RESULTS")
print("--------------------------------")
print(f"MAE:  {float(mae):.4f}")
print(f"RMSE: {float(rmse):.4f}")

metrics_df = pd.DataFrame([{
    "Model": "LSTM",
    "Look_Back_Days": LOOK_BACK,
    "Test_Days": TEST_DAYS,
    "Store_Item_Combinations": len(top_combinations),
    "MAE": float(mae),
    "RMSE": float(rmse)
}])

metrics_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 8. GENERATE NEXT-DAY DEMAND FORECASTS
# ============================================================

print("\nGenerating next-day forecasts...")

next_day_records = []

for metadata in series_metadata:

    store = metadata["store"]
    item = metadata["item"]
    scale = metadata["scale"]

    combo_df = df[
        (df["store"] == store) &
        (df["item"] == item)
    ]

    daily_sales = (
        combo_df.groupby("date")["sales"]
        .sum()
        .reindex(all_dates, fill_value=0)
    )

    scaled_sales = (
        daily_sales / scale
    ).tolist()

    last_window = scaled_sales[-LOOK_BACK:]

    input_tensor = tf.convert_to_tensor(
        [last_window],
        dtype=tf.float32
    )

    input_tensor = tf.expand_dims(
        input_tensor,
        axis=-1
    )

    forecast_scaled = model.predict(
        input_tensor,
        verbose=0
    )[0][0]

    forecast = max(
        0,
        float(forecast_scaled) * scale
    )

    next_day_records.append({
        "store": store,
        "item": item,
        "forecast_date": all_dates.max() + pd.Timedelta(days=1),
        "predicted_demand": forecast
    })

next_day_df = pd.DataFrame(next_day_records)

next_day_path = os.path.join(
    OUTPUT_DIR,
    "next_day_forecast.csv"
)

next_day_df.to_csv(
    next_day_path,
    index=False
)

print("\nNext-day demand forecasts:")
print(next_day_df.head(10))


# ============================================================
# 9. CREATE ACTUAL VS PREDICTED SVG PLOT
# ============================================================

print("\nCreating prediction plot...")

# Plot one representative store-item combination.
first_store = top_combinations[0][0]
first_item = top_combinations[0][1]

plot_df = predictions_df[
    (predictions_df["store"] == first_store) &
    (predictions_df["item"] == first_item)
].sort_values("date")

if len(plot_df) > 0:

    width = 900
    height = 450

    margin_left = 70
    margin_right = 30
    margin_top = 40
    margin_bottom = 60

    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    values = (
        plot_df["actual_sales"].tolist() +
        plot_df["predicted_sales"].tolist()
    )

    max_value = max(values) if values else 1
    max_value = max(max_value, 1) * 1.1

    def get_x(index):
        if len(plot_df) <= 1:
            return margin_left

        return (
            margin_left +
            index * chart_width / (len(plot_df) - 1)
        )

    def get_y(value):
        return (
            margin_top +
            chart_height -
            (value / max_value) * chart_height
        )

    actual_points = []
    predicted_points = []

    for index, row in enumerate(
        plot_df.itertuples(index=False)
    ):

        x = get_x(index)

        actual_points.append(
            f"{x:.2f},{get_y(row.actual_sales):.2f}"
        )

        predicted_points.append(
            f"{x:.2f},{get_y(row.predicted_sales):.2f}"
        )

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{width}" height="{height}"
         viewBox="0 0 {width} {height}">

      <rect width="100%" height="100%" fill="white"/>

      <text x="{width/2}" y="25"
            text-anchor="middle"
            font-size="18" font-family="Arial">
        Actual vs Predicted Demand
        (Store {first_store}, Item {first_item})
      </text>

      <line x1="{margin_left}" y1="{margin_top}"
            x2="{margin_left}" y2="{height-margin_bottom}"
            stroke="black"/>

      <line x1="{margin_left}" y1="{height-margin_bottom}"
            x2="{width-margin_right}" y2="{height-margin_bottom}"
            stroke="black"/>

      <polyline points="{' '.join(actual_points)}"
                fill="none" stroke="#2563eb"
                stroke-width="2.5"/>

      <polyline points="{' '.join(predicted_points)}"
                fill="none" stroke="#dc2626"
                stroke-width="2.5"/>

      <text x="{width/2}" y="{height-15}"
            text-anchor="middle"
            font-size="13" font-family="Arial">
        Test period (days)
      </text>

      <text x="18" y="{height/2}"
            transform="rotate(-90 18 {height/2})"
            text-anchor="middle"
            font-size="13" font-family="Arial">
        Sales
      </text>

      <line x1="{width-250}" y1="45"
            x2="{width-220}" y2="45"
            stroke="#2563eb" stroke-width="3"/>

      <text x="{width-210}" y="49"
            font-size="12" font-family="Arial">
        Actual
      </text>

      <line x1="{width-140}" y1="45"
            x2="{width-110}" y2="45"
            stroke="#dc2626" stroke-width="3"/>

      <text x="{width-100}" y="49"
            font-size="12" font-family="Arial">
        Predicted
      </text>

    </svg>
    """

    plot_path = os.path.join(
        OUTPUT_DIR,
        "actual_vs_predicted.svg"
    )

    with open(plot_path, "w", encoding="utf-8") as file:
        file.write(svg)

    print("Plot saved:", plot_path)


# ============================================================
# 10. SAVE MODEL AND METADATA
# ============================================================

print("\nSaving demand forecasting model...")

model.save(MODEL_PATH)

metadata_df = pd.DataFrame(series_metadata)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "series_metadata.csv"
)

metadata_df.to_csv(
    metadata_path,
    index=False
)

print("\n====================================")
print("DAY 9 COMPLETED SUCCESSFULLY!")
print("====================================")

print("Saved model:", MODEL_PATH)
print("Test predictions:", predictions_path)
print("Evaluation metrics:", metrics_path)
print("Next-day forecast:", next_day_path)
print("Series metadata:", metadata_path)
