import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Define colors
colors = {
    'data': '#E8F4FD',      # Light blue
    'process': '#FFF2CC',   # Light yellow
    'model': '#D5E8D4',     # Light green
    'output': '#F8CECC'     # Light red
}

# Function to create rounded rectangle boxes
def create_box(x, y, width, height, text, color, fontsize=12):
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.2",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=fontsize, fontweight='bold')
    return box

# Function to create arrows
def create_arrow(start_x, start_y, end_x, end_y):
    arrow = ConnectionPatch((start_x, start_y), (end_x, end_y), 
                           "data", "data",
                           arrowstyle="->", shrinkA=10, shrinkB=10,
                           mutation_scale=25, fc="black", linewidth=3)
    ax.add_patch(arrow)

# Title
ax.text(5, 7.5, 'United Airlines Flight Delay Prediction Model', 
        ha='center', va='center', fontsize=18, fontweight='bold')

# 8 Key boxes - technical workflow only
create_box(0.5, 6, 2.5, 0.8, 'Data Loading\n• UA Flight Data\n• Weather Data', colors['data'], 10)
create_box(3.5, 6, 2.5, 0.8, 'Data Cleaning\n• Remove Outliers\n• Handle Missing Values', colors['process'], 10)
create_box(6.5, 6, 2.5, 0.8, 'Feature Engineering\n• 77 Features\n• Time, Weather, Route', colors['process'], 10)

create_box(0.5, 4.5, 2.5, 0.8, 'Data Preparation\n• Split Features\n• Validate Data', colors['process'], 10)
create_box(3.5, 4.5, 2.5, 0.8, 'Preprocessing\n• Scale Numerical\n• Encode Categorical', colors['process'], 10)
create_box(6.5, 4.5, 2.5, 0.8, 'Model Training\n• RandomForest\n• GradientBoosting', colors['model'], 10)

create_box(2, 3, 3, 0.8, 'Model Evaluation\n• Performance Metrics\n• Feature Importance', colors['output'], 10)
create_box(5.5, 3, 3, 0.8, 'Final Prediction\n• Departure Delay\n• R² = 0.739', colors['output'], 10)

# Create arrows connecting the boxes
# Row 1 arrows
create_arrow(3, 6.4, 3.5, 6.4)
create_arrow(6, 6.4, 6.5, 6.4)

# Row 2 arrows
create_arrow(3, 4.9, 3.5, 4.9)
create_arrow(6, 4.9, 6.5, 4.9)

# Vertical arrows connecting rows
create_arrow(1.75, 6, 1.75, 4.8)
create_arrow(4.75, 6, 4.75, 4.8)
create_arrow(7.75, 6, 7.75, 4.8)

# Row 3 arrows
create_arrow(3, 3.4, 3.5, 3.4)
create_arrow(5, 3.4, 5.5, 3.4)

# Add legend
legend_elements = [
    patches.Patch(color=colors['data'], label='Data Sources'),
    patches.Patch(color=colors['process'], label='Processing Steps'),
    patches.Patch(color=colors['model'], label='Machine Learning'),
    patches.Patch(color=colors['output'], label='Results & Outputs')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
          fontsize=10, frameon=True, fancybox=True, shadow=True)

# Add input/output labels
ax.text(0.5, 7.2, 'INPUT', ha='center', va='center', fontsize=12, fontweight='bold', 
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
ax.text(9.5, 7.2, 'OUTPUT', ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcoral', alpha=0.8))

plt.tight_layout()
plt.savefig('united_airlines_8box_flow.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("8-box model flow diagram saved as 'united_airlines_8box_flow.png'")
print("\n8-Step Technical Workflow:")
print("1. Data Loading → 2. Data Cleaning → 3. Feature Engineering")
print("4. Data Preparation → 5. Preprocessing → 6. Model Training")
print("7. Model Evaluation → 8. Final Prediction")
print("Result: Predict departure delays with 73.9% accuracy") 