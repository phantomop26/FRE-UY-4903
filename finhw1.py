import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

data_aapl = pd.read_csv("AAPL.csv")  
data_ba = pd.read_csv("BA.csv") 


data_aapl["Dt"] = pd.to_datetime(data_aapl["Dt"])
data_ba["Dt"] = pd.to_datetime(data_ba["Dt"])

data_aapl = data_aapl.sort_values("Dt").reset_index(drop=True)
data_ba = data_ba.sort_values("Dt").reset_index(drop=True)

data_aapl["Return"] = data_aapl["Close"].pct_change()
data_ba["Return"] = data_ba["Close"].pct_change()

data_2018_aapl = data_aapl[(data_aapl["Dt"] >= "2018-01-01") & (data_aapl["Dt"] <= "2018-12-31")].dropna()
data_2018_ba = data_ba[(data_ba["Dt"] >= "2018-01-01") & (data_ba["Dt"] <= "2018-12-31")].dropna()

num_returns_aapl = len(data_2018_aapl)
first_return_aapl = data_2018_aapl["Return"].iloc[0]
last_return_aapl = data_2018_aapl["Return"].iloc[-1]
avg_return_aapl = data_2018_aapl["Return"].mean()

print("AAPL: There are {num:d} returns. First={first:3.2f}, Last={last:3.2f}, Avg={avg:3.2f}".format(
    num=num_returns_aapl, first=first_return_aapl, last=last_return_aapl, avg=avg_return_aapl))

train_data_aapl = data_2018_aapl[(data_2018_aapl["Dt"].dt.month >= 1) & (data_2018_aapl["Dt"].dt.month <= 9)]
test_data_aapl = data_2018_aapl[(data_2018_aapl["Dt"].dt.month >= 10) & (data_2018_aapl["Dt"].dt.month <= 12)]

train_num_returns_aapl = len(train_data_aapl)
train_first_return_aapl = train_data_aapl["Return"].iloc[0]
train_last_return_aapl = train_data_aapl["Return"].iloc[-1]
train_avg_return_aapl = train_data_aapl["Return"].mean()

test_num_returns_aapl = len(test_data_aapl)
test_first_return_aapl = test_data_aapl["Return"].iloc[0]
test_last_return_aapl = test_data_aapl["Return"].iloc[-1]
test_avg_return_aapl = test_data_aapl["Return"].mean()

print("Training set (AAPL): There are {num:d} returns. First={first:3.2f}, Last={last:3.2f}, Avg={avg:3.2f}".format(
    num=train_num_returns_aapl, first=train_first_return_aapl, last=train_last_return_aapl, avg=train_avg_return_aapl))

print("Test set (AAPL): There are {num:d} returns. First={first:3.2f}, Last={last:3.2f}, Avg={avg:3.2f}".format(
    num=test_num_returns_aapl, first=test_first_return_aapl, last=test_last_return_aapl, avg=test_avg_return_aapl))

spy_data = pd.read_csv("SPY.csv")
spy_data["Dt"] = pd.to_datetime(spy_data["Dt"])
spy_data = spy_data.sort_values("Dt").reset_index(drop=True)
spy_data["Return"] = spy_data["Close"].pct_change()
spy_2018 = spy_data[(spy_data["Dt"] >= "2018-01-01") & (spy_data["Dt"] <= "2018-12-31")].dropna()

merged_data_aapl = pd.merge(data_2018_aapl, spy_2018, on="Dt", suffixes=("_AAPL", "_SPY"))

X_aapl = merged_data_aapl[["Return_SPY"]].values
y_aapl = merged_data_aapl["Return_AAPL"].values

X_aapl = np.c_[np.ones(X_aapl.shape[0]), X_aapl]

model_aapl = LinearRegression(fit_intercept=False)
model_aapl.fit(X_aapl, y_aapl)

beta_0_aapl = model_aapl.coef_[0]
beta_SPY_aapl = model_aapl.coef_[1]

print("AAPL: beta_0={b0:3.2f}, beta_SPY={b1:3.2f}".format(b0=beta_0_aapl, b1=beta_SPY_aapl))

cross_val_avg_aapl = np.mean(cross_val_score(model_aapl, X_aapl, y_aapl, cv=5))

print("AAPL: Avg cross val score = {sc:3.2f}".format(sc=cross_val_avg_aapl))

merged_data_aapl["Hedged_Return_AAPL"] = merged_data_aapl["Return_AAPL"] - beta_SPY_aapl * merged_data_aapl["Return_SPY"]

hedged_num_returns_aapl = len(merged_data_aapl)
hedged_first_return_aapl = merged_data_aapl["Hedged_Return_AAPL"].iloc[0]
hedged_last_return_aapl = merged_data_aapl["Hedged_Return_AAPL"].iloc[-1]
hedged_avg_return_aapl = merged_data_aapl["Hedged_Return_AAPL"].mean()

print("AAPL hedged returns: There are {num:d} returns. First={first:3.2f}, Last={last:3.2f}, Avg={avg:3.2f}".format(
    num=hedged_num_returns_aapl, first=hedged_first_return_aapl, last=hedged_last_return_aapl, avg=hedged_avg_return_aapl))

merged_data_ba = pd.merge(data_2018_ba, spy_2018, on="Dt", suffixes=("_BA", "_SPY"))

X_ba = merged_data_ba[["Return_SPY"]].values
y_ba = merged_data_ba["Return_BA"].values

X_ba = np.c_[np.ones(X_ba.shape[0]), X_ba]

model_ba = LinearRegression(fit_intercept=False)
model_ba.fit(X_ba, y_ba)

beta_0_ba = model_ba.coef_[0]
beta_SPY_ba = model_ba.coef_[1]

print("BA: beta_0={b0:3.2f}, beta_SPY={b1:3.2f}".format(b0=beta_0_ba, b1=beta_SPY_ba))

cross_val_avg_ba = np.mean(cross_val_score(model_ba, X_ba, y_ba, cv=5))

print("BA: Avg cross val score = {sc:3.2f}".format(sc=cross_val_avg_ba))

merged_data_ba["Hedged_Return_BA"] = merged_data_ba["Return_BA"] - beta_SPY_ba * merged_data_ba["Return_SPY"]

hedged_num_returns_ba = len(merged_data_ba)
hedged_first_return_ba = merged_data_ba["Hedged_Return_BA"].iloc[0]
hedged_last_return_ba = merged_data_ba["Hedged_Return_BA"].iloc[-1]
hedged_avg_return_ba = merged_data_ba["Hedged_Return_BA"].mean()

print("BA hedged returns: There are {num:d} returns. First={first:3.2f}, Last={last:3.2f}, Avg={avg:3.2f}".format(
    num=hedged_num_returns_ba, first=hedged_first_return_ba, last=hedged_last_return_ba, avg=hedged_avg_return_ba))

actual_prices_aapl = test_data_aapl["Close"].values

predicted_prices_aapl = [actual_prices_aapl[0]]

for i in range(1, len(test_data_aapl)):
    predicted_price = predicted_prices_aapl[i - 1] * (1 + test_data_aapl["Return"].iloc[i])
    predicted_prices_aapl.append(predicted_price)

num_prices = len(predicted_prices_aapl)
first_price = predicted_prices_aapl[0]
last_price = predicted_prices_aapl[-1]
avg_price = np.mean(predicted_prices_aapl)

ticker = "AAPL"
print("{t:s} predicted prices: There are {num:d} prices. First={first:3.2f}, Last={last:3.2f}, Avg={avg:3.2f}".format(
    t=ticker, num=num_prices, first=first_price, last=last_price, avg=avg_price))





# Extra credits
data_xlb = pd.read_csv('XLB.csv')
data_xlb["Dt"] = pd.to_datetime(data_xlb["Dt"])
data_xlb = data_xlb.sort_values("Dt").reset_index(drop=True)
data_xlb["Return"] = data_xlb["Close"].pct_change()
data_2018_xlb = data_xlb[(data_xlb["Dt"] >= "2018-01-01") & (data_xlb["Dt"] <= "2018-12-31")].dropna()
