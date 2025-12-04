import matplotlib.pyplot as plt
import numpy as np

models = ['gpt-oss-20b', 'Llama-3.1-8B', 'Qwen3-VL-8B', 'Reference', 'Randomized']
accuracy = [0.5927, 0.3639, 0.4235, 0.4609, 0.4150]
spearman = [0.6165, 0.0839, 0.3174, 0.0494, -0.1104]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, accuracy, width, label='Accuracy', color='steelblue')
ax.bar(x + width/2, spearman, width, label='Spearman', color='coral')

ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend()
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('performance.png', dpi=300, bbox_inches='tight')
plt.show()