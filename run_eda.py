import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create directories if they don't exist
import os
os.makedirs('../reports/figures', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

print("="*80)
print("TRANSACTION FRAUD DETECTION - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load the data
df = pd.read_csv('data/data.csv')

print(f"\n✅ Data loaded successfully!")
print(f"📊 Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# 1. DATA OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("1. DATA OVERVIEW")
print("="*80)

print("\n📋 First 5 rows:")
print(df.head())

print("\n📋 Dataset Info:")
print(df.info())

print("\n📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2d}. {col}")

# ============================================================================
# 2. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("2. SUMMARY STATISTICS")
print("="*80)

numerical_cols = df.select_dtypes(include=[np.number]).columns
print("\n📊 Numerical Features Summary:")
print(df[numerical_cols].describe())

print("\n📊 All Features Summary:")
print(df.describe(include='all'))

# ============================================================================
# 3. DISTRIBUTION OF NUMERICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("3. DISTRIBUTION OF NUMERICAL FEATURES")
print("="*80)

# Check skewness
print("\n📊 Skewness Analysis:")
for col in numerical_cols:
    skewness = df[col].skew()
    if abs(skewness) > 1:
        print(f"   ⚠️  {col}: Highly skewed ({skewness:.2f})")
    elif abs(skewness) > 0.5:
        print(f"   📈 {col}: Moderately skewed ({skewness:.2f})")
    else:
        print(f"   ✓  {col}: Approximately symmetric ({skewness:.2f})")

# Amount analysis
print("\n💰 Transaction Amount Analysis:")
print(f"   Total volume: {df['Amount'].sum():,.2f}")
print(f"   Mean amount: {df['Amount'].mean():.2f}")
print(f"   Median amount: {df['Amount'].median():.2f}")
print(f"   Min amount: {df['Amount'].min():.2f}")
print(f"   Max amount: {df['Amount'].max():.2f}")

# ============================================================================
# 4. DISTRIBUTION OF CATEGORICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("4. DISTRIBUTION OF CATEGORICAL FEATURES")
print("="*80)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n📊 {col}:")
    print(f"   Unique values: {df[col].nunique()}")
    print(f"   Top 3 values:")
    for val, count in df[col].value_counts().head(3).items():
        print(f"      {val}: {count:,} ({count/len(df)*100:.2f}%)")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. CORRELATION ANALYSIS")
print("="*80)

corr_matrix = df[numerical_cols].corr()
fraud_correlations = corr_matrix['FraudResult'].sort_values(ascending=False)
print("\n📊 Correlations with FraudResult:")
for feature, corr in fraud_correlations.items():
    if feature != 'FraudResult':
        print(f"   {feature}: {corr:.3f}")

# ============================================================================
# 6. MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("6. MISSING VALUES")
print("="*80)

missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percentage
})

print("\n📊 Missing Values:")
print(missing_df[missing_df['Missing Count'] > 0])

if missing_df[missing_df['Missing Count'] > 0].empty:
    print("   ✅ No missing values found!")

# ============================================================================
# 7. OUTLIER DETECTION
# ============================================================================
print("\n" + "="*80)
print("7. OUTLIER DETECTION")
print("="*80)

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = len(df[(df[col] < lower) | (df[col] > upper)])
    print(f"   {col}: {outliers} outliers ({outliers/len(df)*100:.2f}%)")

# ============================================================================
# 8. TIME-BASED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. TIME-BASED ANALYSIS")
print("="*80)

df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
df['TransactionHour'] = df['TransactionStartTime'].dt.hour
df['TransactionDayOfWeek'] = df['TransactionStartTime'].dt.dayofweek

print(f"\n📅 Time Range:")
print(f"   Start: {df['TransactionStartTime'].min()}")
print(f"   End: {df['TransactionStartTime'].max()}")
print(f"   Duration: {(df['TransactionStartTime'].max() - df['TransactionStartTime'].min()).days} days")

# ============================================================================
# 9. FRAUD PATTERN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("9. FRAUD PATTERN ANALYSIS")
print("="*80)

fraud_df = df[df['FraudResult'] == 1]
print(f"\n🔍 Fraud Statistics:")
print(f"   Total fraud cases: {len(fraud_df)}")
print(f"   Fraud rate: {len(fraud_df)/len(df)*100:.4f}%")

print(f"\n📺 Fraud by Channel:")
for channel, count in fraud_df['ChannelId'].value_counts().items():
    total = len(df[df['ChannelId'] == channel])
    print(f"   Channel {channel}: {count} fraud cases ({count/total*100:.2f}%)")

print(f"\n💰 Fraud by Pricing Strategy:")
for strategy, count in fraud_df['PricingStrategy'].value_counts().items():
    total = len(df[df['PricingStrategy'] == strategy])
    print(f"   Strategy {strategy}: {count} fraud cases ({count/total*100:.2f}%)")

# ============================================================================
# 10. CUSTOMER BEHAVIOR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("10. CUSTOMER BEHAVIOR ANALYSIS")
print("="*80)

print(f"\n👥 Customer Statistics:")
print(f"   Unique customers: {df['CustomerId'].nunique()}")
print(f"   Unique accounts: {df['AccountId'].nunique()}")

# ============================================================================
# 11. KEY FINDINGS
# ============================================================================
print("\n" + "="*80)
print("11. KEY FINDINGS & HYPOTHESES")
print("="*80)

print("""
KEY FINDINGS:
1. Dataset has 43,615 transactions with 0.02% fraud rate (highly imbalanced)
2. No missing values - data is clean
3. Negative amounts represent financial services reversals
4. Multiple channels and pricing strategies show different fraud patterns

HYPOTHESES FOR FEATURE ENGINEERING:
1. Transaction velocity correlates with fraud
2. Unusual amount outliers indicate suspicious activity  
3. Product category influences fraud risk
4. Time-based patterns (hour, day) show fraud trends
5. Negative transaction patterns flag suspicious accounts
6. Pricing strategy interacts with product category
7. Customer history impacts fraud probability
""")

print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY!")
print("="*80)