import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Load and Prepare Data
data = pd.read_csv('E:\Benjamin\Startups.csv')  # Update the path as needed
X = data[['R&D Spend', 'Administration', 'Marketing Spend']]
y = data['Profit']

# 2. Divide the data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Construct Different Regression Algorithms
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Support Vector Regression
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)

# 4. Calculate Different Regression Metrics
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, R^2: {r2:.2f}")
    return mse, r2

mse_lr, r2_lr = print_metrics(y_test, y_pred_lr, 'Linear Regression')
mse_rf, r2_rf = print_metrics(y_test, y_pred_rf, 'Random Forest')
mse_svr, r2_svr = print_metrics(y_test, y_pred_svr, 'Support Vector Regression')

# 5. Choose the Best Model
best_model = max(
    [('Linear Regression', r2_lr, mse_lr), 
     ('Random Forest', r2_rf, mse_rf), 
     ('Support Vector Regression', r2_svr, mse_svr)],
    key=lambda x: x[1]  # Maximize R^2
)

print(f"Best Model: {best_model[0]} with R^2: {best_model[1]:.2f} and MSE: {best_model[2]:.2f}")
