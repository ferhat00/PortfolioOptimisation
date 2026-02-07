# Portfolio Optimization & Stock Price Prediction

A comprehensive Python-based repository for portfolio optimization and stock price prediction using Modern Portfolio Theory (MPT), machine learning, and deep learning techniques.

## 📋 Overview

This project combines two major areas of quantitative finance:

1. **Portfolio Optimization** - Implementing various optimization strategies to construct efficient portfolios
2. **Stock Price Prediction** - Using machine learning models to forecast stock price movements

The repository provides Jupyter notebooks for both educational purposes and practical implementation, allowing users to:
- Download and analyze historical stock data
- Build optimized portfolios using different risk-return objectives
- Train and evaluate ML models for price prediction
- Visualize portfolio performance and financial metrics

## 🎯 Key Features

### Portfolio Optimization
- **Mean-Variance Optimization** - Classic Markowitz portfolio theory
- **Maximum Sharpe Ratio** - Optimize for best risk-adjusted returns
- **Minimum Variance** - Construct lowest-risk portfolios
- **Target Volatility** - Achieve specific risk levels
- **Efficient Frontier** - Visualize risk-return trade-offs
- **Backtesting** - Test portfolio strategies on historical data

### Machine Learning for Price Prediction
- **Multiple Algorithms**: 
  - Deep Neural Networks (Keras/TensorFlow)
  - Random Forest
  - LightGBM (with and without cross-validation)
  - Logistic Regression
- **Technical Indicators** - Feature engineering using price and volume data
- **Model Evaluation** - Performance metrics and visualization
- **Walk-forward Analysis** - Time-series aware validation

## 📁 Repository Structure

### Main Notebooks

#### Portfolio Optimization
- **[Portfolio_optimisation.ipynb](Portfolio_optimisation.ipynb)** - Core portfolio optimization implementation using MPT
- **[Portfolio_Optimisation_SIPP_v2.ipynb](Portfolio_Optimisation_SIPP_v2.ipynb)** - Advanced optimization with multiple strategies (max Sharpe, min variance, target volatility, manual weights)
- **[Portfolio_optimisation_SIPP.ipynb](Portfolio_optimisation_SIPP.ipynb)** - Earlier version of SIPP optimization
- **[Plot_Portfolio.ipynb](Plot_Portfolio.ipynb)** - Portfolio visualization and performance plotting tools
- **[riskfolio_example.ipynb](riskfolio_example.ipynb)** - Example implementations using the Riskfolio library
- **[Tutorial 6_unique.ipynb](Tutorial 6_unique.ipynb)** - Tutorial on portfolio optimization concepts

#### Machine Learning Price Prediction
- **[Price_Prediction_ML_v2.ipynb](Price_Prediction_ML_v2.ipynb)** - Main ML price prediction pipeline
- **[Keras_DNN.ipynb](Keras_DNN.ipynb)** - Deep Neural Network implementation using Keras
- **[Random_Forest.ipynb](Random_Forest.ipynb)** - Random Forest classifier/regressor for price prediction
- **[light_gbm.ipynb](light_gbm.ipynb)** - LightGBM gradient boosting model
- **[light_gbm-cross_val.ipynb](light_gbm-cross_val.ipynb)** - LightGBM with cross-validation
- **[Logistic_Regression.ipynb](Logistic_Regression.ipynb)** - Logistic Regression for directional prediction

## 🛠️ Dependencies

### Core Libraries
```python
# Data manipulation and analysis
numpy
pandas
scipy
statsmodels

# Financial data
yfinance
pandas_datareader
yahoofinancials

# Visualization
matplotlib
mplfinance
seaborn
plotly

# Portfolio optimization
pypfopt
riskfolio-lib
quantstats

# Machine Learning
scikit-learn
tensorflow
keras
lightgbm

# Utilities
ray
timebudget
pickle
```

## 🚀 Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ferhat00/PortfolioOptimisation.git
cd PortfolioOptimisation
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install numpy pandas scipy statsmodels
pip install yfinance pandas-datareader yahoofinancials
pip install matplotlib mplfinance seaborn plotly
pip install PyPortfolioOpt riskfolio-lib quantstats
pip install scikit-learn tensorflow lightgbm
pip install ray timebudget
```

### Quick Start

#### Portfolio Optimization Example
```python
# Define your stock universe
stocks = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
start_date = '2010-01-01'
end_date = '2024-01-01'

# Run Portfolio_optimisation.ipynb or Portfolio_Optimisation_SIPP_v2.ipynb
# to construct an optimized portfolio
```

#### Price Prediction Example
```python
# Choose a stock
stock = ['SPY']
date_start = '1993-01-01'

# Run any of the ML notebooks to train and evaluate models
# - Price_Prediction_ML_v2.ipynb for comprehensive pipeline
# - Keras_DNN.ipynb for deep learning approach
# - light_gbm.ipynb for gradient boosting
```

## 📊 Workflow

### 1. Data Collection
- Notebooks use Yahoo Finance API to download historical price data
- Support for daily, weekly, and monthly sampling
- Automatic data preprocessing and cleaning

### 2. Portfolio Optimization
```
Data Download → Calculate Returns → Estimate Covariance → 
Optimize Weights → Backtest → Evaluate Performance
```

Key optimization objectives:
- **Max Sharpe**: Maximize risk-adjusted returns
- **Min Variance**: Minimize portfolio volatility
- **Target Volatility**: Achieve specific risk level
- **Manual Weights**: Custom allocations

### 3. Price Prediction
```
Data Download → Feature Engineering → Train/Test Split → 
Model Training → Evaluation → Prediction
```

Models available:
- **Neural Networks**: Multi-layer perceptrons with dropout
- **Tree-based**: Random Forest, LightGBM
- **Linear**: Logistic Regression

## 📈 Key Metrics & Outputs

### Portfolio Optimization
- **Sharpe Ratio** - Risk-adjusted return metric
- **Sortino Ratio** - Downside risk-adjusted return
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Volatility** - Standard deviation of returns
- **Cumulative Returns** - Total portfolio performance
- **Efficient Frontier** - Risk-return visualization
- **Portfolio Weights** - Optimal asset allocations

### Price Prediction
- **Accuracy** - Classification accuracy for directional prediction
- **Precision/Recall** - Model prediction quality
- **RMSE/MAE** - Regression error metrics
- **ROC Curve** - Model discrimination ability
- **Feature Importance** - Key predictive variables
- **Predicted vs Actual** - Visualization of model performance

## 🎓 Use Cases

1. **Academic Research** - Study portfolio theory and ML in finance
2. **Investment Strategy Development** - Design and backtest portfolio strategies
3. **Risk Management** - Analyze portfolio risk characteristics
4. **Algorithmic Trading** - Build predictive models for trading signals
5. **Financial Education** - Learn about quantitative finance techniques

## ⚙️ Configuration

Most notebooks include configuration sections at the top:

```python
# Portfolio Optimization Settings
stock = ['SPY', 'GLD', 'QQQ', 'TLT']
date_start = '2010-01-01'
date_end = '2024-01-01'
max_sharpe = True
min_variance = True
target_vol = False

# ML Settings
sampling = 'daily'  # or 'weekly'
train_start = '2010-01-01'
train_end = '2021-12-01'
test_start = '2021-12-02'
test_end = date.today()
```

## 📝 Notable Features

### Portfolio_Optimisation_SIPP_v2.ipynb
- Comprehensive optimization framework
- Multiple optimization objectives
- L2 regularization support
- Semi-variance and CVaR options
- Quantstats integration for detailed analytics
- Historical bear/bull market dates reference

### Price_Prediction_ML_v2.ipynb
- Complete ML pipeline
- Feature engineering with technical indicators
- Multiple model comparison
- Walk-forward validation
- Extensive visualizations

## 🔬 Technical Details

### Data Sources
- **Yahoo Finance** - Primary data source via `yfinance` and `pandas_datareader`
- Support for stocks, ETFs, indices, commodities, futures
- Global market coverage (US, UK, Europe, Asia)

### Optimization Methods
- **Convex Optimization** - Using cvxpy via pypfopt
- **Covariance Estimation** - Ledoit-Wolf shrinkage, sample covariance
- **Expected Returns** - Historical mean, CAPM, custom estimates

### ML Techniques
- **Cross-validation** - Time-series split to prevent look-ahead bias
- **Hyperparameter Tuning** - Grid search, random search
- **Feature Scaling** - StandardScaler, MinMaxScaler
- **Regularization** - L1/L2 for preventing overfitting

## ⚠️ Important Notes

1. **Past Performance**: Historical returns do not guarantee future results
2. **Data Quality**: Results depend on data accuracy and completeness
3. **Market Conditions**: Models trained on specific periods may not generalize
4. **Transaction Costs**: Backtests do not include trading fees and slippage
5. **Rebalancing**: Portfolio optimization assumes periodic rebalancing
6. **API Limitations**: Yahoo Finance API has rate limits and occasional data gaps

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional optimization algorithms
- More ML models
- Enhanced risk metrics
- Real-time data integration
- Transaction cost modeling
- Multi-period optimization

## 📄 License

This project is available for educational and research purposes.

## 📧 Contact

For questions or collaboration:
- GitHub: [@ferhat00](https://github.com/ferhat00)

## 🙏 Acknowledgments

This project uses several excellent open-source libraries:
- **PyPortfolioOpt** - Portfolio optimization
- **Riskfolio-Lib** - Risk-based portfolio optimization
- **QuantStats** - Portfolio analytics
- **LightGBM** - Gradient boosting framework
- **Keras/TensorFlow** - Deep learning

## 📚 References

### Portfolio Theory
- Markowitz, H. (1952). "Portfolio Selection"
- Sharpe, W. (1964). "Capital Asset Prices: A Theory of Market Equilibrium"

### Machine Learning in Finance
- Advances in Financial Machine Learning (Marcos López de Prado)
- Machine Learning for Asset Managers (Marcos López de Prado)

---

**Last Updated**: February 2026

**Status**: Active Development

**Python Version**: 3.8+
