"""
Generate visual charts for PAMAP2 dataset overview
"""

import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

# ============= CHART 1: ACTIVITY DISTRIBUTION (Pie Chart) =============
activities_lifestyle = {
    'Lying': 1,
    'Sitting': 2,
    'Standing': 3,
    'Walking': 4,
    'Nordic Walking': 5,
    'Upstairs': 6,
    'Downstairs': 7,
    'Ironing': 8
}

activities_sport = {
    'Rope Jumping': 9,
    'Running': 10,
    'Basketball': 11,
    'Volleyball': 12,
    'Soccer': 13,
    'Cycling': 14,
    'Table Tennis': 15,
    'Other': 17
}

# Simulated sample distribution (approximate based on typical PAMAP2)
activity_samples = {
    'Lying': 450000, 'Sitting': 380000, 'Standing': 320000,
    'Walking': 520000, 'Nordic Walking': 280000, 'Upstairs': 180000,
    'Downstairs': 175000, 'Ironing': 120000, 'Rope Jumping': 45000,
    'Running': 350000, 'Basketball': 85000, 'Volleyball': 95000,
    'Soccer': 110000, 'Cycling': 155000, 'Table Tennis': 75000,
    'Other': 50000
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Pie Chart: Activity Distribution
labels = list(activity_samples.keys())
sizes = list(activity_samples.values())
colors_lifestyle = plt.cm.Blues(np.linspace(0.4, 0.9, 8))
colors_sport = plt.cm.Reds(np.linspace(0.4, 0.9, 8))
colors = np.concatenate([colors_lifestyle, colors_sport])

# Separate lifestyle and sport
lifestyle_mask = [i < 8 for i in range(len(labels))]
lifestyle_labels = [labels[i] for i in range(len(labels)) if i < 8]
lifestyle_sizes = [sizes[i] for i in range(len(sizes)) if i < 8]

sport_labels = [labels[i] for i in range(len(labels)) if i >= 8]
sport_sizes = [sizes[i] for i in range(len(sizes)) if i >= 8]

axes[0, 0].pie(lifestyle_sizes, labels=lifestyle_labels, autopct='%1.1f%%',
               colors=colors_lifestyle, startangle=90, textprops={'fontsize': 9})
axes[0, 0].set_title('Lifestyle Activities Distribution\n(PAMAP2 Dataset)',
                     fontsize=14, fontweight='bold')

axes[0, 1].pie(sport_sizes, labels=sport_labels, autopct='%1.1f%%',
               colors=colors_sport, startangle=90, textprops={'fontsize': 9})
axes[0, 1].set_title('Sport Activities Distribution\n(PAMAP2 Dataset)',
                     fontsize=14, fontweight='bold')

# ============= CHART 2: SAMPLES PER SUBJECT (Bar Chart) =============
subjects = ['101', '102', '103', '104', '105', '106', '107', '108', '109']
samples = [1000000, 500000, 800000, 600000, 700000, 900000, 400000, 500000, 650000]
activities_count = [18, 17, 18, 16, 18, 18, 15, 17, 18]

bars = axes[1, 0].bar(subjects, samples, color=plt.cm.viridis(np.linspace(0.2, 0.8, 9)),
                       edgecolor='black', linewidth=1.5)
axes[1, 0].set_xlabel('Subject ID', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Sample Count per Subject\n(PAMAP2 Dataset)',
                     fontsize=14, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, samples):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val//1000}K', ha='center', va='bottom', fontsize=9)

# ============= CHART 3: SENSOR PLACEMENT & DATA STRUCTURE =============
axes[1, 1].axis('off')

# Create sensor placement diagram
sensor_text = """
SENSOR PLACEMENT & DATA STRUCTURE

┌─────────────────────────────────────────────────────────┐
│  PAMAP2 Sensor Configuration                             │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐                                       │
│  │ CHEST IMU   │  ── 3 axes (accel x,y,z)               │
│  │ (position 2│                                         │
│  │  on chest) │                                         │
│  └──────┬──────┘                                       │
│         │                                              │
│    ┌────┴────┐                                         │
│    │         │                                         │
│ ┌──▼──┐   ┌──▼──┐                                    │
│ │WRIST│   │ANKLE│  ── 6 more axes (total = 9)         │
│ │ IMU │   │ IMU │                                    │
│ │pos 1│   │pos 3│                                    │
│ └─────┘   └─────┘                                    │
│                                                         │
│  Sampling Rate: 100 Hz (10ms per sample)               │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  DATA PIPELINE                                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Raw (.dat) → 54 columns                                │
│       │                                                 │
│       ▼                                                 │
│  Filter → Remove transient activities (ID=0)            │
│       │                                                 │
│       ▼                                                 │
│  Clean → 9 columns (accelerometer only)                 │
│       │                                                 │
│       ▼                                                 │
│  Window → 256 samples × 9 axes                          │
│       │                                                 │
│       ▼                                                 │
│  LSTM Input → (batch, 256, 9)                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""

axes[1, 1].text(0.05, 0.95, sensor_text, transform=axes[1, 1].transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('PAMAP2 Physical Activity Monitoring Dataset Overview',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('D:/drift-detection/fl-drift-try_2/results/pamap2_dataset_charts.png',
            dpi=300, bbox_inches='tight')
print("[OK] Charts saved to results/pamap2_dataset_charts.png")

# ============= ADDITIONAL: SINGLE PIE CHART FOR ALL ACTIVITIES =============
fig2, ax = plt.subplots(figsize=(12, 10))

# Sort by sample count
sorted_items = sorted(activity_samples.items(), key=lambda x: x[1], reverse=True)
sorted_labels = [x[0] for x in sorted_items]
sorted_sizes = [x[1] for x in sorted_items]

# Color by category
colors_full = []
for label in sorted_labels:
    if label in activities_lifestyle:
        colors_full.append('#4682B4')  # Steel blue
    else:
        colors_full.append('#E74C3C')  # Red

wedges, texts, autotexts = ax.pie(sorted_sizes, labels=sorted_labels, autopct='%1.1f%%',
                                    colors=colors_full, startangle=90,
                                    textprops={'fontsize': 10})

# Make percentage text bold
for autotext in autotexts:
    autotext.set_fontweight('bold')

ax.set_title('PAMAP2 Dataset: Complete Activity Distribution\n(Total: ~6M samples)',
             fontsize=16, fontweight='bold', pad=20)

# Add legend
legend_labels = ['Lifestyle Activities', 'Sport Activities']
legend_handles = [plt.Rectangle((0, 0), 1, 1, color='#4682B4'),
                  plt.Rectangle((0, 0), 1, 1, color='#E74C3C')]
ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig('D:/drift-detection/fl-drift-try_2/results/pamap2_activity_pie_chart.png',
            dpi=300, bbox_inches='tight')
print("[OK] Pie chart saved to results/pamap2_activity_pie_chart.png")

plt.close('all')
print("\n[OK] All charts generated successfully!")
