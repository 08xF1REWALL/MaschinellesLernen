# Training Simple Machine Learning Algorithms for Classification
## Preceptron 

 **Dokumentation:**  
[PDF öffnen](docs/Chapter2_preceptron.pdf)

- OvR: One-vs-Rest is a technique that allows us to extend a binary classifier per class, where the particular class is treated as positive class and the samples from all other classes as negative class.


1. Was ist "target" im Code und im Perceptron?Einfache Definition: target ist das wahre Label (auch "Ground Truth" oder "Zielwert" genannt) für ein einzelnes Trainingsbeispiel. Es ist der korrekte Klassifikationswert, den das Modell lernen soll.Im Perceptron-Code (in der fit-Methode): target kommt aus dem Array y (den Labels, die du dem Modell gibst).
Im Code: for xi, target in zip(X, y): – Hier wird target für jedes Sample xi zugewiesen. Es ist also y(i)y^{(i)}y^{(i)}
 für das i-te Beispiel.
Werte: Im Perceptron sind Labels binär: -1 oder +1 (nicht 0/1, weil die Mathematik mit -1/+1 einfacher ist – der Fehler wird dann -2/0/+2).

Im Iris-Datensatz-Beispiel (aus deinem Code):Wir haben zwei Klassen: Iris-Setosa und Iris-Versicolor.
y = np.where(y == 'Iris-setosa', -1, 1): Wenn die Blume Setosa ist → target = -1.
Wenn Versicolor ist → target = +1.

Warum -1/+1? Das ist Konvention im Perceptron-Algorithmus, um die Update-Regel einfach zu halten (Fehler = y - ŷ ist dann symmetrisch).

Zusammenhang zum Update: In der Regel update = self.eta * (target - self.predict(xi)) ist target der "richtige" Wert, mit dem du die Vorhersage y^\hat{y}\hat{y}
 (aus self.predict(xi)) vergleichst. Wenn sie gleich sind, update=0 (keine Änderung). Wenn nicht, korrigiert es die Gewichte.

2. Detaillierte Erklärung deines BeispielsatzesDu hast den Satz zitiert: "Wenn target = -1 (falsch): Update = 0.1 * (-1 - (+1)) = 0.1 * (-2) = -0.2."Kontext des Beispiels: Ich habe ein fiktives Sample erfunden, um zu zeigen, wie das Update funktioniert.Sample: xi = [5.0, 1.5] (Sepal Length=5 cm, Petal Length=1.5 cm – typisch für Setosa, die kleine Blütenblätter hat).
Angenommene Gewichte: w_ = [-0.5, 0.2, 0.8] (w_0=Bias=-0.5, w_1=0.2, w_2=0.8).
Berechnung: net_input z = 1.7 (>=0) → predict(xi) = +1 (Vorhersage: Versicolor).

Target = -1: Das bedeutet: Das wahre Label für dieses Sample ist -1 (es ist wirklich eine Setosa-Blume).
Warum "(falsch)"? Weil die Vorhersage y^=+1\hat{y} = +1\hat{y} = +1
 ist, aber target = -1 – also hat das Modell einen Fehler gemacht (falsch klassifiziert). Das Wort "falsch" bezieht sich auf die Vorhersage im Vergleich zum target.
Wenn target = -1 und ŷ = +1: Fehler! Das Modell denkt, es ist Versicolor (+1), aber es ist Setosa (-1).

Update-Berechnung:eta = 0.1 (Lernrate).
update = 0.1 * (target - ŷ) = 0.1 * (-1 - (+1)) = 0.1 * (-2) = -0.2.
Warum -2? Weil y = -1, ŷ = +1 → y - ŷ = -1 - 1 = -2.
Das negative Update (-0.2) zieht die Gewichte "in die Richtung" von Setosa, um zukünftige Vorhersagen für ähnliche Samples zu korrigieren (z wird negativer).

Was passiert danach?Gewichte-Update: self.w_[1:] += -0.2 * xi → Die Gewichte werden kleiner (negativ angepasst).
Bias: self.w_[0] += -0.2 → Bias wird negativer.
Ergebnis: Die Entscheidungsgrenze verschiebt sich, damit dieses Sample nächstes Mal korrekt als -1 klassifiziert wird.

3. Was, wenn target = +1?Beispiel: Angenommen target = +1 (wahres Label: Versicolor), aber ŷ = -1 (Vorhersage falsch).Update = 0.1 * (+1 - (-1)) = 0.1 * (+2) = +0.2.
Positives Update: Gewichte werden größer, um z positiver zu machen.

Wenn target == ŷ (z. B. target=-1, ŷ=-1): Update = 0.1 * ( -1 - (-1) ) = 0.1 * 0 = 0 → Kein Update (korrekt).

4. Warum ist target wichtig?Target ist der "Lehrer"-Wert: Es sagt dem Modell, was richtig ist. Ohne target könnte das Modell nicht lernen (das ist Supervised Learning).
Im Code: y (das Array mit allen targets) kommt aus den Daten (z. B. Iris-Labels).

Falls du das immer noch nicht ganz verstehst, gib mir mehr Details: Welchen Teil von "target" meinst du genau (z. B. warum -1/+1, oder im Iris-Beispiel)? Ich kann ein weiteres Beispiel mit realen Iris-Daten rechnen oder den Code anpassen, um targets auszugeben. Lass es mich wissen!

## Adaptive Linear Neurons
