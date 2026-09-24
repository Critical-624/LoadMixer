# LoadMixer

LoadMixer is a PyTorch model for forecasting Prefill and Decode token workloads in LLM serving.
Provide a CSV file to train the model, evaluate it, or forecast future values.

## Install

Requires Python 3.10 or later and Git.

```bash
git clone https://github.com/Critical-624/LoadMixer.git
cd LoadMixer
python -m pip install -r requirements.txt
```

## Data

Use a CSV with a `date` column and a numeric target column:

```csv
date,load
2026-01-01 00:00:00,1200
2026-01-01 00:10:00,1350
2026-01-01 00:20:00,1280
```

Supply the full series in chronological order with equally spaced timestamps
and no missing values. The rows above illustrate the format; training requires
a longer series. Replace `load` below with your target column name.

## Train and test

```bash
python run.py --data_path path/to/workload.csv --target load
```

This trains the model and evaluates the held-out test split, saving
`checkpoint.pt` and `metrics.json` under `outputs/loadmixer/`.
Use `--output_dir` to choose a new directory for another training run.
The chronological train/validation/test split is 70/10/20, with normalization
fitted on training data and early stopping based on validation loss.
MAE, MSE and RMSE are reported in original and standardized units,
averaged over forecast windows and steps.

To evaluate the saved model on the same CSV:

```bash
python run.py --mode test --data_path path/to/workload.csv --checkpoint outputs/loadmixer/checkpoint.pt
```

## Forecast

Provide recent observations using the same columns and time interval:

```bash
python run.py --mode predict --data_path path/to/recent.csv --checkpoint outputs/loadmixer/checkpoint.pt
```

Future timestamps and predictions are saved to `outputs/loadmixer/forecast.csv`.
Run `python run.py --help` for options, or `python example.py` for a model-only example.

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
