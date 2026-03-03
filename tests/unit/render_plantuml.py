import subprocess
import os

# Pfade anpassen, falls plantuml.jar woanders liegt
plantuml_jar = "plantuml.jar"  # oder vollständiger Pfad
puml_file = "adaline_aktivitaetsdiagramm.puml"
output_dir = "..\\..\\src\\main\\images"

os.makedirs(output_dir, exist_ok=True)

try:
    subprocess.run([
        "java", "-jar", plantuml_jar,
        "-o", output_dir,
        puml_file
    ], check=True)
    print(f"Diagramm erfolgreich erstellt: {output_dir}\\adaline_aktivitaetsdiagramm.png")
except FileNotFoundError:
    print("Java oder plantuml.jar nicht gefunden. Alternativen:")
    print("1. VS Code Extension 'PlantUML' installieren und Alt+D drücken")
    print("2. Online-Renderer: http://www.plantuml.com/plantuml")
except subprocess.CalledProcessError as e:
    print(f"Fehler beim Rendern: {e}")
