#Ajeet kumar
#import modules:
import pandas as pd;import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
# importing data:
data = pd.read_csv("daily_covid_cases.csv")
test_size = 0.35  # 35% for testing:
X = data.values
tst_sz = math.ceil(len(X) * test_size)
train, test = X[:len(X) - tst_sz], X[len(X) - tst_sz:]
train = pd.DataFrame(train, columns=["Date", "New_cases"])
test = pd.DataFrame(test, columns=["Date", "New_cases"])
train.to_csv("train.csv", index=None)
def autoreg_model(train_, test_, lag_value):  # Autoreg model returns actual, predicted and coefficients values:
    window = lag_value  # Lag Value:
    model = AR(np.array(train_["New_cases"]), lags=window)
    model_fit = model.fit()
    coef = model_fit.params
    print("Lag value = %i" % window)
    history = list(train_["New_cases"][len(train_["New_cases"])-window:])
    predictions = list()
    for t in range(len(np.array(test_["New_cases"]))):
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        yhat = coef[0]  # Initialize to w0
        for d in range(window):
            yhat += coef[d + 1] * lag[window - d - 1]  # Add other values
        predictions.append(yhat)  # Append predictions to compute RMSE
        obs = np.array(test_["New_cases"])[t]
        history.append(obs)  # Append actual test value to history, to be
    return history[:len(test_)], predictions, [float("%.3f" % i) for i in coef]
print("-Question_1-")
#Q
fig, ax = plt.subplots()
ax.plot(data["Date"], data["new_cases"])
date_form = DateFormatter("%b-%d")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.set(xlabel="Month-Year",
ylabel="Number of Covid-19 Cases")
plt.title("Daily Covid-19 Cases From 30Jan2020 to 2Oct2021", loc='left')
plt.xticks(rotation=45)
plt.show()
#Q
t = data["new_cases"].shift(1)
data["lagged_new_cases"] = t
#Q
plt.scatter(data["new_cases"].iloc[1:], data["lagged_new_cases"].iloc[1:])
plt.xlabel("Given Time series data")
plt.ylabel("Lagged Time Series data")
plt.title("Scatter plot between Given and Lagged Data", loc='left')
plt.show()
print("--Question_1_part_c and d--")
lag = [i for i in range(1, 7)]
corr_values = []  # correlation values
for i in lag:
    data["new_lag_series"] = data['new_cases'].shift(i)
    corr = data["new_cases"].iloc[i:].corr(data["new_lag_series"].iloc[i:])
    print("Pearson Correlation for %i day lagged time series =" % i, "%.3f" % corr)
    corr_values.append(corr)
plt.plot(lag, corr_values, color="#EE2C2C")
plt.scatter(lag, corr_values, 100)
plt.xlabel("Lagged Values")
plt.ylabel("Correlation Value")
plt.title("Lagged Value Vs Correlation ", loc="left")
for i in range(len(lag)): # adding text in the plot:
    plt.text(x=i+1.05, y=corr_values[i], s="%.3f" % corr_values[i], size=9)
plt.show()
#Q1
sm.graphics.tsa.plot_acf(data["new_cases"], lags=7)  # plot using inbuilt function:
plt.xlabel("Lag Values")
plt.ylabel("Correlation Coefficient")
plt.show()
#Q2
print("-Q_2_a-")
# Actual, predicted and coefficients respectively:
actual, prediction, coef = autoreg_model(train, test, 5)
print("Coefficient of Autorig Model")
print(coef)
print("-Q_2_b-")
# Scatter plot between actual and predicted data:
test["Actual Data"] = actual
test["Predicted Data"] = prediction
plt.scatter(test["Actual Data"], test["Predicted Data"], color="#EE2C2C")
plt.xlabel("Actual Data")
plt.ylabel("Predicted Data")
plt.title("Actual Vs Predicted Data for lag = 5", loc="left")
plt.show()
# line plot between actual and predicted data:
fig, ax = plt.subplots()
ax.plot(test["Date"], test["Actual Data"], label="Actual Data")
ax.plot(test["Date"], test["Predicted Data"], label="Predicted Data")
date_form = DateFormatter("%b-%d")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.set(xlabel="Month-Year",
ylabel="Number of Covid-19 Cases")
plt.title("Daily Covid-19 Cases From 30Jan2020 to 2Oct2021", loc='left')
plt.xticks(rotation=45)
plt.legend()
plt.show()
# RMSE and MAPE error:
rmse = mean_squared_error(test["Actual Data"], test["Predicted Data"], squared=False)
mape = mean_absolute_percentage_error(test["Actual Data"], test["Predicted Data"])
print("RMSE between actual and predicted test data", "%.3f" % (rmse/test["Actual Data"].mean() * 100), "%")
print("MAPE between actual and predicted test data", "%.3f" % mape)
#Q3
print("-Question_3-")
RMSE = []
MAPE = []
# print(np.array(train[1]))
lag_value = [1, 5, 10, 15, 25]
for g in lag_value:
    actual, prediction, coef = autoreg_model(train, test, g)
    test["Actual Data"] = actual
    test["Predicted Data"] = prediction
    rmse = mean_squared_error(test["Actual Data"], test["Predicted Data"], squared=False)
    mape = mean_absolute_percentage_error(test["Actual Data"], test["Predicted Data"])
    RMSE.append((rmse / test["Actual Data"].mean() * 100))
    MAPE.append(mape)
    print("RMSE between actual and predicted test data", "%.3f" % (rmse*len(test["Actual Data"]) / test["Actual Data"].sum() * 100), "%")
    print("MAPE between actual and predicted test data", "%.3f" % mape)
# bar plot of different RMSE Values:
plt.bar(["1", "5", "10", "15", "25"], RMSE, width=0.4, color="#009ACD")
plt.xlabel("Value of lag")
plt.ylabel("Value of RMSE in percentage")
plt.title("Lag Value Vs RMSE", loc="left")
plt.show()
# bar plot of different MAPE Values:
plt.bar(["1", "5", "10", "15", "25"], MAPE, width=0.4,  color="#008B45")
plt.xlabel("Value of lag")
plt.ylabel("Value of MAPE")
plt.title("Lag Value Vs MAPE", loc="left")
plt.show()
print()
print("-Question_4-")
train = pd.read_csv("train.csv")
# Optimal Lag value:
i = 0
corr = 1
# abs(AutoCorrelation) > 2/sqrt(T)
while corr > 2/(len(train))**0.5:
    i += 1
    train["new_lag_series"] = train["New_cases"].shift(i)
    corr = train.corr()["New_cases"][1]
print("Optimal Value of Lag is")
actual, prediction, coef = autoreg_model(train, test, i)
rmse = mean_squared_error(actual, prediction, squared=False)
mape = mean_absolute_percentage_error(actual, prediction)
print("RMSE between actual and predicted test data",
        "%.3f" % (rmse * len(actual) / np.array(actual).sum() * 100), "%")
print("MAPE between actual and predicted test data", "%.3f" % mape)

