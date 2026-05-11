nithin
lang+ english

# --- NithinLang V1 Machine Learning Demo ---

print("=== NithinLang ML Demo ===")

# 1. Create arrays
X = np_array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
              [2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])
y = np_array([0, 0, 0, 1, 1, 1])

print("Feature matrix shape:", len(X), "rows")

# 2. Train a classifier
model = model_train(X, y, "random_forest", test_size=0.3)

# 3. Predict
preds = model_predict(model, [[2.5, 2.5], [5.0, 5.0]])
print("Predictions:", preds)

# 4. NumPy operations
arr  = np_linspace(0, 10, 100)
mean = np_mean(arr)
std  = np_std(arr)
print("Mean:", mean, "  Std:", std)

# 5. Save & load model
model_save(model, "rf_model.pkl")
loaded = model_load("rf_model.pkl")
print("Loaded model:", loaded)

# 6. K-Means clustering
result  = cluster_fit(X, k=2)
print("Cluster labels:", result["labels"])
print("Cluster centres:", result["centers"])

# 7. Timing
timer_start("loop")
total = 0
for i in range(1_000_000):
    total = total + i
elapsed = timer_stop("loop")
print("Sum:", total, "  Time:", elapsed, "s")

end nithin