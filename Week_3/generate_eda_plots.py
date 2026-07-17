import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set plotting style for professional look
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True

# Load dataset
data_path = '../Week_2/Steel_industry_data/Steel_industry_data.csv'
if not os.path.exists(data_path):
    # Adjust path if script is run from project root instead of Week_3 folder
    data_path = 'Week_2/Steel_industry_data/Steel_industry_data.csv'

df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M', errors='coerce')
df['Hour'] = df['date'].dt.hour
df['DayOfWeek'] = df['date'].dt.dayofweek
df['Month'] = df['date'].dt.month
df['IsWeekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
df['Power_Factor_Ratio'] = df['Leading_Current_Power_Factor'] / df['Lagging_Current_Power_Factor']
df['Power_Factor_Ratio'] = df['Power_Factor_Ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Output directory
output_dir = 'Week_3/static/images'
if not os.path.exists(output_dir):
    # If run inside Week_3 directory
    output_dir = 'static/images'
os.makedirs(output_dir, exist_ok=True)

# 1. Visualization 1: Energy Consumption by Hour
plt.figure(figsize=(10, 5))
hourly_avg = df.groupby('Hour')['Usage_kWh'].mean().reset_index()
sns.lineplot(data=hourly_avg, x='Hour', y='Usage_kWh', marker='o', color='#2d6a4f', linewidth=2.5) # Forest Green
plt.title('Average Energy Consumption (kWh) by Hour of Day', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Hour of Day (0-23)', fontsize=12)
plt.ylabel('Average Usage (kWh)', fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energy_by_hour.png'), dpi=150)
plt.close()
print("Saved energy_by_hour.png")

# 2. Visualization 2: Energy Consumption by Load Type
plt.figure(figsize=(9, 5.5))
load_summary = df.groupby('Load_Type')['Usage_kWh'].mean().reset_index()
load_order = ['Light_Load', 'Medium_Load', 'Maximum_Load']
load_summary['Load_Type'] = pd.Categorical(load_summary['Load_Type'], categories=load_order, ordered=True)
load_summary = load_summary.sort_values('Load_Type')

colors = ['#d6ccc2', '#38bdf8', '#2d6a4f']  # Warm Beige, Sky Blue, and Forest Green
ax = sns.barplot(x='Load_Type', y='Usage_kWh', data=load_summary, palette=colors, edgecolor='black', linewidth=1.2)

# Annotate bars with values
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f} kWh", 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 8), 
                textcoords='offset points', 
                fontsize=11, fontweight='bold')

plt.title('Average Energy Consumption by Load Type', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Load Type', fontsize=12)
plt.ylabel('Average Usage (kWh)', fontsize=12)
plt.xticks(ticks=range(3), labels=['Light Load', 'Medium Load', 'Maximum Load'])
plt.ylim(0, load_summary['Usage_kWh'].max() * 1.15)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'energy_by_load_type.png'), dpi=150)
plt.close()
print("Saved energy_by_load_type.png")

# 3. Visualization 3: Correlation Heatmap
plt.figure(figsize=(11, 9))
num_cols = [
    'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 
    'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 
    'NSM', 'Hour', 'DayOfWeek', 'Power_Factor_Ratio'
]
corr_matrix = df[num_cols].corr()

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='YlGnBu', vmin=-1, vmax=1, linewidths=0.5, square=True) # YlGnBu Colormap
plt.title('Correlation Heatmap of Numerical Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150)
plt.close()
print("Saved correlation_heatmap.png")
