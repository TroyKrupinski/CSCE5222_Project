import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def add_box(ax, text, xy, width=3, height=1, color='lightblue', fontsize=10):
    # Create a rounded rectangle with text
    rect = patches.FancyBboxPatch((xy[0] - width / 2, xy[1] - height / 2),
                                  width, height,
                                  boxstyle="round,pad=0.1",
                                  edgecolor='black',
                                  facecolor=color)
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], text, ha='center', va='center', fontsize=fontsize)

def add_arrow(ax, start, end):
    # Add an arrow between two points
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15)
ax.axis('off')

# Define steps and positions
steps = [
    ("Image Loading and Preprocessing", (5, 14), 'skyblue'),
    ("Load Image\nConvert to Grayscale\nHistogram Equalization\nNoise Reduction", (5, 12.5), 'lightblue'),
    ("Component Detection", (5, 11), 'skyblue'),
    ("Multi-Scale Detection\nThresholding\nMorphological Operations\nConnected Components", (5, 9.5), 'lightblue'),
    ("Feature Extraction", (5, 8), 'skyblue'),
    ("Shape Features\nIntensity Features\nTexture Features", (5, 6.5), 'lightblue'),
    ("Classification (SVM)", (5, 5), 'skyblue'),
    ("Data Splitting\nSVM Training\nPrediction and Metrics", (5, 3.5), 'lightblue'),
    ("Visualization", (5, 2), 'skyblue'),
    ("Display Images and Detected Components\nShow Component Properties", (5, 0.5), 'lightblue')
]

# Draw boxes and text
for text, pos, color in steps:
    add_box(ax, text, pos, color=color)

# Draw arrows between steps
for i in range(len(steps) - 1):
    add_arrow(ax, steps[i][1], steps[i + 1][1])

plt.show()
