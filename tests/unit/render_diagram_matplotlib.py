import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, ax = plt.subplots(figsize=(8, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')

# Boxen und Positionen
steps = [
    ("CSV-Daten laden (Iris)", 5, 19),
    ("Auswahl der ersten 100 Samples\n(Setosa vs Versicolor)", 5, 17.5),
    ("Features (sepal_length, petal_length)", 5, 16),
    ("Labels erzeugen (-1 / +1)", 5, 14.5),
    ("Standardisierung der Features", 5, 13),
    ("Figure mit 2 Subplots erstellen", 5, 11.5),
    ("Training (eta = 0.01):\nGewichte init, Loop über Epochen,\nnet_input, Fehler, Update, Kosten", 5, 9),
    ("log(cost) plotten", 5, 7),
    ("Training (eta = 0.0001):\nGleicher Ablauf", 5, 5.5),
    ("cost plotten", 5, 4),
    ("Diagramm speichern & anzeigen", 5, 2.5),
]

for text, x, y in steps:
    rect = mpatches.FancyBboxPatch(
        (x - 2, y - 0.4), 4, 0.8,
        boxstyle="round,pad=0.1", 
        edgecolor='black', facecolor='lightblue', linewidth=1.5
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, weight='bold')

# Pfeile zwischen Schritten
for i in range(len(steps) - 1):
    _, x0, y0 = steps[i]
    _, x1, y1 = steps[i + 1]
    ax.annotate('', xy=(x1, y1 + 0.4), xytext=(x0, y0 - 0.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

plt.title("Aktivitätsdiagramm – AdalineGD Training", fontsize=14, weight='bold')
plt.tight_layout()

output_dir = "..\\..\\src\\main\\images"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "adaline_aktivitaetsdiagramm.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Diagramm gespeichert: {output_path}")
plt.show()
