# LoadMixer

PyTorch implementation of **Forecasting LLM Serving Workloads via Multiscale Learning and Prediction Modulation**.

## Overview

LoadMixer forecasts prefill and decode token demand in LLM serving as separate univariate forecasting tasks. It combines multiscale representation learning with adaptive prediction modulation to capture local bursts and changing workload conditions.

The model consists of three components:

- **Multiscale temporal representation:** convolutional downsampling captures position-dependent local variations, while last-value centering references recent workload levels.
- **Cross-scale feature mixing:** trend and seasonal components are mixed across temporal resolutions.
- **Adaptive prediction modulation:** state-guided fusion adjusts scale contributions, and multi-window difference correction supplements predictions with recent local changes.

The implementation uses 96 historical observations and predicts 12 future steps by default. With 10-minute workload intervals, this corresponds to 16 hours of history and a 2-hour forecast.

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/Critical-624/LoadMixer.git
cd LoadMixer
python -m pip install -r requirements.txt
```

## Data Preparation

Prepare a CSV containing a timestamp column named `date` and one or more numeric workload columns. For example:

```csv
date,num_prefill_tokens,num_decode_tokens
2026-01-01 00:00:00,1200,2400
2026-01-01 00:10:00,1350,2700
2026-01-01 00:20:00,1280,2560
```

Rows must be in chronological order, with equally spaced timestamps and no missing values. The rows above illustrate the format; training requires a longer series.

Use `--target` to select the workload column. Prefill and decode models are trained separately. Other column names are also supported.

## Training and Evaluation

Train a model for prefill token demand:

```bash
python run.py \
  --data_path datasets/workload.csv \
  --target num_prefill_tokens \
  --loss MAE \
  --output_dir outputs/prefill
```

Train a separate model for decode token demand:

```bash
python run.py \
  --data_path datasets/workload.csv \
  --target num_decode_tokens \
  --loss MAE \
  --output_dir outputs/decode
```

The runner splits the series chronologically into training, validation, and test sets with a 70/10/20 ratio. Standardization uses training-set statistics, and the best checkpoint is selected by validation loss with early stopping.

After training, the model is evaluated on the test split. Each output directory contains:

| File | Contents |
|---|---|
| `checkpoint.pt` | Model weights, configuration, and preprocessing statistics |
| `metrics.json` | Test MAE, MSE, and RMSE in original and standardized units |

To evaluate a saved checkpoint on the same dataset:

```bash
python run.py \
  --mode test \
  --data_path datasets/workload.csv \
  --checkpoint outputs/prefill/checkpoint.pt \
  --output_dir outputs/prefill
```

## Forecasting

Provide a CSV with at least 96 recent observations, using the same target column and sampling interval as the training data:

```bash
python run.py \
  --mode predict \
  --data_path datasets/recent.csv \
  --checkpoint outputs/prefill/checkpoint.pt \
  --output_dir outputs/prefill
```

Future timestamps and predictions in original workload units are saved to `outputs/prefill/forecast.csv`.

The forecasting horizon can be configured with `--pred_len` during training. For all available options:

```bash
python run.py --help
```

## Code Structure

| File | Purpose |
|---|---|
| `loadmixer.py` | LoadMixer model |
| `layers.py` | Shared model layers |
| `data.py` | CSV loading, calendar features, and forecasting windows |
| `run.py` | Training, evaluation, and forecasting |
| `example.py` | Model usage example |

## Acknowledgements

This implementation includes code adapted from [TimeMixer](https://github.com/kwuking/TimeMixer). We thank its authors for making their implementation publicly available.

## License

This project is licensed under Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
