# NeuronDB Installation Guide

## Quick Install

```bash
./build.sh
```

This automated script will:
1. Install all ML library prerequisites (XGBoost, LightGBM, CatBoost)
2. Build the NeuronDB extension
3. Show installation status

## Manual Installation

### Prerequisites

#### macOS
```bash
brew install xgboost lightgbm
pip3 install catboost
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install build-essential cmake git

# XGBoost
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install

# LightGBM  
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM && mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install

# CatBoost
git clone https://github.com/catboost/catboost.git
cd catboost/catboost/libs && mkdir build && cd build
cmake .. -DCATBOOST_BUILD_LIBRARY=ON && make -j$(nproc)
sudo make install
```

### Build NeuronDB

```bash
PG_CONFIG=/path/to/pg_config make
sudo PG_CONFIG=/path/to/pg_config make install
```

## Conditional Compilation

NeuronDB uses conditional compilation for ML libraries:

- **XGBoost**: Detected via `__has_include(<xgboost/c_api.h>)`
- **LightGBM**: Detected via `__has_include(<LightGBM/c_api.h>)` 
- **CatBoost**: Detected via `__has_include(<catboost/c_api.h>)`

If libraries are not found, NeuronDB will:
- ✅ Build successfully without errors
- ✅ Provide graceful error messages when functions are called
- ✅ Work with all other ML algorithms (12 built-in algorithms always available)

## Verify Installation

```sql
CREATE EXTENSION neurondb;

-- Test XGBoost (requires libxgboost)
SELECT train_xgboost_classifier('table', 'features', 'label');

-- Test built-in algorithms (always available)
SELECT train_linear_regression('table', 'features', 'target');
SELECT train_random_forest_classifier('table', 'features', 'label', 10, 10, 100);
```

## Troubleshooting

### XGBoost not found
```bash
# macOS
brew install xgboost

# Linux
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### Build errors
```bash
# Clean and rebuild
make clean
./build.sh
```

## Support

- Built-in ML (always available): 12 algorithms
- External ML (optional): XGBoost, LightGBM, CatBoost
- GPU Support: CUDA, ROCm, Metal (Apple Silicon)

