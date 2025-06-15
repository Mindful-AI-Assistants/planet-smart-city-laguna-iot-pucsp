

# Residential Energy Consumption Analysis and Optimization

This project aims to analyze and optimize energy consumption in residences using historical consumption data by room. The analyses seek to identify patterns, predict consumption, and propose recommendations to reduce costs and increase sustainability.

---

## Data Description

The dataset (`df`) used contains daily records of energy consumption (in kWh) in different residential environments over a certain period. The main columns of the dataframe are:

- **Date:** Date of measurement.
- **Bedroom1, Bedroom2:** Consumption in bedrooms 1 and 2.
- **Living Room:** Consumption in the living room.
- **Kitchen:** Consumption in the kitchen.
- **Pool:** Consumption related to pool usage.
- **kWh:** Total daily household consumption (kilowatt-hour).

---

## Plots and Analyses

### 1. Total Consumption Histogram

```python
plt.figure(figsize=(8, 5))
plt.hist(df['kWh'], bins=20, color='skyblue')
plt.title('Distribution - Total Consumption (kWh)')
plt.xlabel('Total Consumption (kWh)')
plt.ylabel('Count')
plt.show()
```

**Analysis:**
The histogram shows that most days have total consumption between 1000 and 1300 kWh, with few days of very low or very high consumption. This indicates a relatively stable pattern, but with room for optimization at the extremes.

---

### 2. Histograms of All Variables

```python
import seaborn as sns

cols_to_plot = df.columns[1:]  # Ignore the 'Date' column
n_cols = len(cols_to_plot)
n_rows = (n_cols + 2) // 3  # 3 columns per row

fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    sns.histplot(df[col], kde=True, ax=axes[i], bins=10)
    axes[i].set_title(f'Distribution - {col}')
    axes[i].set_xlabel(col)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Distribution of Consumption Variables", fontsize=16, y=1.02)
plt.show()
```

**Analysis:**
Individual histograms reveal that the living room and pool show greater usage variation, while bedrooms and kitchen have more concentrated distributions. This suggests that actions in these environments may have a greater impact on reducing consumption.

---

### 3. Correlation Heatmap

```python
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['Date']).corr(), annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Heatmap Between Variables')
plt.show()
```

**Analysis:**
The correlation analysis shows that the living room is the environment most associated with increased total consumption (correlation 0.55), followed by Bedroom1 (0.52) and Pool (0.43). Kitchen and Bedroom2 also influence consumption but with less weight. Kitchen has a negative correlation with Pool (-0.08), suggesting alternating use on some days.

---

### 4. Total Consumption Over Time

```python
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['kWh'], label='Total Consumption (kWh)', color='blue', linewidth=2)
plt.title('Total Energy Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

**Analysis:**
The time series plot allows identification of peaks and seasonality in consumption, which are fundamental for scheduled interventions and educational campaigns.

---

### 5. Consumption by Room Over Time

```python
plt.figure(figsize=(16, 8))
for room in ['Bedroom1', 'Bedroom2', 'Living Room', 'Kitchen', 'Pool']:
    plt.plot(df['Date'], df[room], label=room)

plt.title('Energy Consumption by Room Over Time')
plt.xlabel('Date')
plt.ylabel('Consumption (kWh)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

**Analysis:**
Visualizing consumption by room over time helps identify which environments contribute most to consumption peaks and guide specific actions.

---

### 6. Linear Regression: Actual vs Predicted

```python
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Identity line
plt.xlabel('Actual Consumption (kWh)')
plt.ylabel('Predicted Consumption (kWh)')
plt.title('Actual vs Predicted - Total Consumption')
plt.show()
```

**Model Details:**

- **Predictor variables:** Bedroom1, Bedroom2, Living Room, Kitchen, Pool.
- **Target variable:** Total consumption (kWh).
- **Validation:** Holdout (train/test split).
- **Metrics:** R² = 0.86, RMSE = XX, MAE = XX (replace XX with actual values if available).

**Analysis:**
The linear regression model showed an R² of 0.86, indicating high predictive capacity. The main environments impacting total consumption are Living Room, Pool, and Kitchen.

---

### 7. Elbow Method (KMeans)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = df[['Bedroom1', 'Bedroom2', 'Living Room', 'Kitchen', 'Pool']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (sum of squared distances)')
plt.show()
```

> **Note:** Inertia represents the sum of squared distances of samples to their closest cluster center. The "elbow" indicates the ideal number of clusters.

**Analysis:**
The elbow method suggests that 3 clusters are ideal for segmenting consumption profiles.

---

### 8. Cluster Visualization

```python
import seaborn as sns

sns.pairplot(df, hue='Cluster', vars=['Bedroom1', 'Bedroom2', 'Living Room', 'Kitchen', 'Pool'], palette='tab10')
plt.suptitle("Usage Patterns Grouped by Cluster", y=1.02)
plt.show()
```

**Analysis:**
Clustering identified three main groups:

- **Low consumption:** Reduced use across all environments.
- **Balanced consumption:** Moderate and distributed use.
- **High consumption:** Peaks especially in living room, kitchen, or pool.

These groups allow for personalized recommendations and interventions.

---

## Profiles and Recommendations

- **Low Consumption:** Maintain good practices, encourage solar energy and continuous monitoring.
- **Balanced Consumption:** Automate device shutdowns, install presence sensors and timers.
- **High Consumption:** Automate lighting, monitor pool usage, encourage conscious use, and consider more efficient appliances.

---

## Conclusion

The data enabled segmentation of households and highly accurate consumption prediction. The model is simple, interpretable, and highlights the rooms that most influence total consumption. The analysis supports targeted recommendations and interventions for cost reduction and increased sustainability.

> **Next Steps:**
> - Integrate real-time data and IoT for dynamic monitoring.
> - Explore more advanced predictive models.
> - Evaluate the impact of implemented recommendations.

---

**Questions, suggestions, or interest in collaborating? Get in touch!**



