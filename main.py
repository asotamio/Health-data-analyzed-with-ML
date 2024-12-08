# 必要なライブラリをインポート
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# データの読み込み
file_path = '/Users/tamio/Downloads/oura_2024-10-02_2024-12-08_trends.csv'
health_data = pd.read_csv(file_path)

# 数値データのみを選択
numeric_data = health_data.select_dtypes(include=['int64', 'float64'])

# 特徴量（X）と目的変数（y）の分離
X = numeric_data.drop(columns=['Readiness Score'])
y = numeric_data['Readiness Score']

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特徴量のスケーリング
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ランダムフォレストモデルのトレーニング
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train_scaled, y_train)

# テストセットでの予測
y_pred = model.predict(X_test_scaled)

# モデルの評価
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# 重要な特徴量の可視化
feature_importances = model.feature_importances_
important_features = sorted(zip(X.columns, feature_importances), key=lambda x: x[1], reverse=True)

# 上位10特徴量を表示
print("Top 10 Important Features:")
for feature, importance in important_features[:10]:
    print(f"{feature}: {importance:.2f}")

# 特徴量の重要度をプロット
top_features = important_features[:10]
features, importances = zip(*top_features)
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.yticks(fontsize=8) 
plt.show()