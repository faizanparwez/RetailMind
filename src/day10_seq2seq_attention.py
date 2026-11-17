
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import layers, Model


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "../data/train.csv"

MODEL_DIR = "../models"
OUTPUT_DIR = "../outputs/day10"

SEQ2SEQ_PATH = os.path.join(
    MODEL_DIR,
    "demand_seq2seq_attention.keras"
)

BASELINE_PATH = os.path.join(
    MODEL_DIR,
    "day10_single_step_baseline.keras"
)

LOOK_BACK = 30
FORECAST_HORIZON = 7
TEST_DAYS = 28
TOP_N = 20

EPOCHS = 30
BATCH_SIZE = 64

tf.keras.utils.set_random_seed(42)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD CSV USING PYTHON'S BUILT-IN CSV MODULE
# ============================================================

print("\nLoading demand dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Place train.csv inside RetailMind/data/"
    )

sales_data = defaultdict(float)
combination_totals = defaultdict(float)
all_dates = set()

with open(DATA_PATH, "r", newline="", encoding="utf-8") as file:

    reader = csv.DictReader(file)

    required = {"date", "store", "item", "sales"}

    if not required.issubset(reader.fieldnames or []):
        raise ValueError(
            "CSV must contain date, store, item and sales columns."
        )

    for row in reader:

        date = datetime.strptime(
            row["date"], "%Y-%m-%d"
        ).date()

        store = int(row["store"])
        item = int(row["item"])
        sales = max(0.0, float(row["sales"]))

        key = (store, item, date)

        sales_data[key] += sales
        combination_totals[(store, item)] += sales

        all_dates.add(date)


dates = sorted(all_dates)

if len(dates) <= LOOK_BACK + TEST_DAYS:
    raise ValueError("Not enough dates for training and testing.")

print("Total dates:", len(dates))
print("Store-item combinations:", len(combination_totals))


# ============================================================
# 2. SELECT TOP STORE-ITEM COMBINATIONS
# ============================================================

top_combinations = sorted(
    combination_totals,
    key=combination_totals.get,
    reverse=True
)[:TOP_N]

print("\nSelected combinations:", len(top_combinations))

# Build a continuous daily calendar.
calendar = []

current_date = dates[0]
last_date = dates[-1]

while current_date <= last_date:
    calendar.append(current_date)
    current_date += timedelta(days=1)

date_to_index = {
    date: index for index, date in enumerate(calendar)
}

cutoff_index = len(calendar) - TEST_DAYS - 1

print("Training ends:", calendar[cutoff_index])
print("Testing starts:", calendar[cutoff_index + 1])
print("Testing ends:", calendar[-1])


# ============================================================
# 3. PREPARE NORMALIZED SERIES
# ============================================================

print("\nPreparing store-item time series...")

series_list = []

for store, item in top_combinations:

    values = []

    for date in calendar:

        value = sales_data.get(
            (store, item, date), 0.0
        )

        values.append(value)

    training_values = values[:cutoff_index + 1]

    scale = max(training_values)

    if scale <= 0:
        scale = 1.0

    scaled_values = [
        value / scale for value in values
    ]

    series_list.append({
        "store": store,
        "item": item,
        "scale": scale,
        "values": values,
        "scaled": scaled_values
    })


# ============================================================
# 4. CREATE SUPERVISED WINDOWS
# ============================================================

print("\nCreating training windows...")

X_train = []
y_train_one = []
y_train_seven = []

for series in series_list:

    values = series["scaled"]

    for i in range(LOOK_BACK, cutoff_index + 1):

        input_window = values[i - LOOK_BACK:i]

        target_one = values[i]

        target_seven = values[
            i:i + FORECAST_HORIZON
        ]

        if len(target_seven) != FORECAST_HORIZON:
            continue

        if i + FORECAST_HORIZON - 1 > cutoff_index:
            continue

        X_train.append(input_window)

        y_train_one.append(target_one)

        y_train_seven.append(target_seven)


if not X_train:
    raise ValueError("No training sequences were generated.")

X_train = tf.convert_to_tensor(
    X_train, dtype=tf.float32
)

y_train_one = tf.convert_to_tensor(
    y_train_one, dtype=tf.float32
)

y_train_seven = tf.convert_to_tensor(
    y_train_seven, dtype=tf.float32
)

# LSTM input shape: (samples, timesteps, features)
X_train = tf.expand_dims(X_train, axis=-1)

# Single-step output: (samples, 1)
y_train_one = tf.expand_dims(y_train_one, axis=-1)

# Seq2Seq output: (samples, 7, 1)
y_train_seven = tf.expand_dims(y_train_seven, axis=-1)

print("Training samples:", X_train.shape[0])
print("Input shape:", X_train.shape)
print("Single-step target shape:", y_train_one.shape)
print("Seven-day target shape:", y_train_seven.shape)


# ============================================================
# 5. TRAIN DAY 10 SINGLE-STEP BASELINE
# ============================================================

print("\nTraining single-step LSTM baseline...")

baseline_model = tf.keras.Sequential([

    layers.Input(
        shape=(LOOK_BACK, 1)
    ),

    layers.LSTM(64),

    layers.Dense(32, activation="relu"),

    layers.Dense(1)

])

baseline_model.compile(
    optimizer="adam",
    loss="mae",
    metrics=["mae"]
)

baseline_model.summary()

baseline_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

baseline_history = baseline_model.fit(
    X_train,
    y_train_one,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[baseline_early_stopping],
    shuffle=True,
    verbose=1
)

print(
    "Baseline epochs:",
    len(baseline_history.history["loss"])
)


# ============================================================
# 6. BUILD SEQ2SEQ MODEL WITH ATTENTION
# ============================================================

print("\nBuilding Seq2Seq attention model...")

encoder_inputs = layers.Input(
    shape=(LOOK_BACK, 1),
    name="encoder_inputs"
)

# Encoder reads the historical sequence.
encoder_lstm = layers.LSTM(
    64,
    return_sequences=True,
    return_state=True,
    name="encoder_lstm"
)

encoder_outputs, state_h, state_c = encoder_lstm(
    encoder_inputs
)

# Repeat the encoder's final state for all future steps.
decoder_inputs = layers.RepeatVector(
    FORECAST_HORIZON,
    name="decoder_input"
)(state_h)

# Decoder receives the encoder states.
decoder_lstm = layers.LSTM(
    64,
    return_sequences=True,
    name="decoder_lstm"
)

decoder_outputs = decoder_lstm(
    decoder_inputs,
    initial_state=[state_h, state_c]
)

# Attention: decoder steps attend to encoder history.
attention_layer = layers.Attention(
    name="attention"
)

context, attention_scores = attention_layer(
    [decoder_outputs, encoder_outputs],
    return_attention_scores=True
)

# Combine decoder outputs and attention context.
combined = layers.Concatenate(
    name="attention_context"
)([decoder_outputs, context])

dense_layer = layers.Dense(
    32,
    activation="relu",
    name="forecast_dense"
)

forecast_features = dense_layer(combined)

output_layer = layers.Dense(
    1,
    name="demand_output"
)

forecast_outputs = output_layer(forecast_features)

# Training model returns only the forecast.
seq2seq_model = Model(
    encoder_inputs,
    forecast_outputs,
    name="Seq2Seq_Attention"
)

# Separate inspection model also returns attention scores.
attention_model = Model(
    encoder_inputs,
    [forecast_outputs, attention_scores],
    name="Seq2Seq_Attention_Inspector"
)

seq2seq_model.compile(
    optimizer="adam",
    loss="mae",
    metrics=["mae"]
)

seq2seq_model.summary()


# ============================================================
# 7. TRAIN SEQ2SEQ MODEL
# ============================================================

print("\nTraining Seq2Seq attention model...")

seq_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

seq_history = seq2seq_model.fit(
    X_train,
    y_train_seven,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[seq_early_stopping],
    shuffle=True,
    verbose=1
)

print(
    "Seq2Seq epochs:",
    len(seq_history.history["loss"])
)


# ============================================================
# 8. CREATE ROLLING TEST FORECASTS
# ============================================================

print("\nGenerating seven-day test forecasts...")

actual_values = []
baseline_values = []
seq2seq_values = []

prediction_records = []

attention_sum = None
attention_count = 0

# Forecast origins are spaced seven days apart.
# Each new origin can use actual observations available
# before that forecast origin.

forecast_origins = list(
    range(
        cutoff_index + 1,
        len(calendar) - FORECAST_HORIZON + 1,
        FORECAST_HORIZON
    )
)

if not forecast_origins:
    raise ValueError("No test forecast origins available.")

for series in series_list:

    store = series["store"]
    item = series["item"]
    scale = series["scale"]

    values = series["scaled"]

    for origin in forecast_origins:

        history_window = values[
            origin - LOOK_BACK:origin
        ]

        if len(history_window) != LOOK_BACK:
            continue

        input_tensor = tf.convert_to_tensor(
            [history_window],
            dtype=tf.float32
        )

        input_tensor = tf.expand_dims(
            input_tensor,
            axis=-1
        )

        # ----------------------------------------------------
        # Recursive one-step baseline
        # ----------------------------------------------------

        recursive_window = list(history_window)

        baseline_forecast = []

        for step in range(FORECAST_HORIZON):

            model_input = tf.convert_to_tensor(
                [recursive_window],
                dtype=tf.float32
            )

            model_input = tf.expand_dims(
                model_input,
                axis=-1
            )

            next_prediction = baseline_model.predict(
                model_input,
                verbose=0
            )[0][0]

            next_prediction = max(
                0.0,
                float(next_prediction)
            )

            baseline_forecast.append(
                next_prediction * scale
            )

            recursive_window = (
                recursive_window[1:] +
                [next_prediction]
            )

        # ----------------------------------------------------
        # Seven-step Seq2Seq forecast
        # ----------------------------------------------------

        seq_prediction, attention_weights = (
            attention_model.predict(
                input_tensor,
                verbose=0
            )
        )

        seq_prediction = seq_prediction[0, :, 0]

        # ----------------------------------------------------
        # Collect attention weights
        # ----------------------------------------------------

        attention_matrix = attention_weights[0]

        if attention_sum is None:
            attention_sum = attention_matrix.copy()
        else:
            attention_sum += attention_matrix

        attention_count += 1

        # ----------------------------------------------------
        # Collect actuals and predictions
        # ----------------------------------------------------

        for step in range(FORECAST_HORIZON):

            target_index = origin + step

            actual = (
                values[target_index] * scale
            )

            baseline_prediction = (
                baseline_forecast[step]
            )

            seq_prediction_value = max(
                0.0,
                float(seq_prediction[step]) * scale
            )

            actual_values.append(actual)

            baseline_values.append(
                baseline_prediction
            )

            seq2seq_values.append(
                seq_prediction_value
            )

            prediction_records.append({
                "store": store,
                "item": item,
                "forecast_origin": calendar[origin],
                "forecast_date": calendar[target_index],
                "forecast_day": step + 1,
                "actual_sales": actual,
                "baseline_prediction": baseline_prediction,
                "seq2seq_prediction": seq_prediction_value
            })


# ============================================================
# 9. EVALUATE BOTH MODELS
# ============================================================

print("\nEvaluating models...")

actual_tensor = tf.convert_to_tensor(
    actual_values,
    dtype=tf.float32
)

baseline_tensor = tf.convert_to_tensor(
    baseline_values,
    dtype=tf.float32
)

seq2seq_tensor = tf.convert_to_tensor(
    seq2seq_values,
    dtype=tf.float32
)


def calculate_metrics(actual, predicted):

    mae = tf.reduce_mean(
        tf.abs(actual - predicted)
    )

    rmse = tf.sqrt(
        tf.reduce_mean(
            tf.square(actual - predicted)
        )
    )

    return float(mae), float(rmse)


baseline_mae, baseline_rmse = calculate_metrics(
    actual_tensor,
    baseline_tensor
)

seq_mae, seq_rmse = calculate_metrics(
    actual_tensor,
    seq2seq_tensor
)

print("\nMODEL COMPARISON")
print("--------------------------------------")

print(
    f"Single-step LSTM | "
    f"MAE: {baseline_mae:.4f} | "
    f"RMSE: {baseline_rmse:.4f}"
)

print(
    f"Seq2Seq Attention | "
    f"MAE: {seq_mae:.4f} | "
    f"RMSE: {seq_rmse:.4f}"
)


# ============================================================
# 10. SAVE COMPARISON AND PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "multi_horizon_predictions.csv"
)

with open(
    prediction_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = list(prediction_records[0].keys())

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(prediction_records)


metrics_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

with open(
    metrics_path,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Forecast_Horizon_Days",
        "MAE",
        "RMSE"
    ])

    writer.writerow([
        "Single-step LSTM",
        FORECAST_HORIZON,
        baseline_mae,
        baseline_rmse
    ])

    writer.writerow([
        "Seq2Seq Attention",
        FORECAST_HORIZON,
        seq_mae,
        seq_rmse
    ])


# ============================================================
# 11. SAVE ATTENTION WEIGHT VISUALIZATION
# ============================================================

print("\nSaving attention visualization...")

if attention_count > 0:

    average_attention = (
        attention_sum / attention_count
    )

    # Average across all seven decoder steps.
    average_attention = tf.reduce_mean(
        average_attention,
        axis=0
    ).numpy()

    width = 900
    height = 250

    cell_width = 22
    start_x = 110
    start_y = 65

    cell_height = 25

    max_attention = max(
        float(max(average_attention)),
        0.000001
    )

    rectangles = []

    for index, weight in enumerate(average_attention):

        intensity = int(
            255 * float(weight) / max_attention
        )

        color = f"rgb( {255-intensity}, {255-intensity}, 255)"

        x = start_x + index * cell_width

        rectangles.append(
            f"""
            <rect x="{x}" y="{start_y}"
                  width="20" height="{cell_height}"
                  fill="{color}" stroke="#777"/>
            <text x="{x+10}" y="{start_y+42}"
                  font-size="10"
                  text-anchor="middle">{index+1}</text>
            """
        )

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{width}" height="{height}">

      <rect width="100%" height="100%" fill="white"/>

      <text x="{width/2}" y="25"
            text-anchor="middle"
            font-size="18" font-family="Arial">
        Average LSTM Attention Weights
      </text>

      <text x="10" y="{start_y+17}"
            font-size="12" font-family="Arial">
        Past day
      </text>

      {''.join(rectangles)}

      <text x="{width/2}" y="155"
            text-anchor="middle"
            font-size="12" font-family="Arial">
        Historical days (1 = oldest, 30 = most recent)
      </text>

    </svg>
    """

    attention_path = os.path.join(
        OUTPUT_DIR,
        "attention_weights.svg"
    )

    with open(
        attention_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(svg)

    print("Attention visualization saved:", attention_path)


# ============================================================
# 12. SAVE MODELS
# ============================================================

print("\nSaving models...")

baseline_model.save(BASELINE_PATH)

seq2seq_model.save(SEQ2SEQ_PATH)

print("\n======================================")
print("DAY 10 COMPLETED SUCCESSFULLY!")
print("======================================")

print("Seq2Seq model:", SEQ2SEQ_PATH)
print("Baseline model:", BASELINE_PATH)
print("Predictions:", prediction_path)
print("Metrics:", metrics_path)

print("\nNext step: Multi-horizon forecasting with GAN/VAE")
