---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: base
  language: python
  name: python3
---

# Lineare Algebra

+++


## 1. Allgemeine Notationen

+++

### 1.1. Vektor

Wir bezeichnen mit $x \in \mathbb{R}^n$ einen Vektor mit $n$ Einträgen, wobei $x_i \in \mathbb{R}$ der $i$-te Eintrag ist:

$$
x = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} \in \mathbb{R}^n
$$


#### Erklärung

Man stellt sich einen **Vektor** wie eine geordnete Liste von Zahlen vor, zum Beispiel wie eine Einkaufsliste, in der jede Zeile etwas anderes bedeutet.

* Der Ausdruck „$x \in \mathbb{R}^n$“ bedeutet: Wir haben eine Liste mit **n Zahlen**.
* Jede dieser Zahlen heißt „Eintrag“ des Vektors.
* „$x_i \in \mathbb{R}$“ heißt: Der $i$-te Eintrag in der Liste ist einfach eine **reelle Zahl**, also eine Zahl mit oder ohne Nachkommastellen (z. B. 3, -1.5, 0).

Die Schreibweise:

$$
x = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix}
$$

zeigt, dass die Zahlen **untereinander** geschrieben sind, wie eine Spalte.


#### Beispiel

Stell dir vor, du gehst einkaufen und willst drei Dinge kaufen: Äpfel, Bananen und Milch. Du schreibst auf:

* 4 Äpfel
* 6 Bananen
* 1 Liter Milch

Das ist wie ein Vektor mit 3 Einträgen:

$$
x = \begin{pmatrix} 4 \\ 6 \\ 1 \end{pmatrix}
$$

Hier ist:

* $x_1 = 4$ (Äpfel),
* $x_2 = 6$ (Bananen),
* $x_3 = 1$ (Milch).

Du hast also einen Vektor in drei Dimensionen – oder kurz gesagt: **eine Liste mit 3 Zahlen**.

---

+++

#### PyTorch Code
Hier zeigen wir, wie man so einen Vektor in PyTorch darstellt und verwendet:

```{code-cell} ipython3
# Wir importieren die PyTorch-Bibliothek, um mit Tensoren (also Vektoren) zu arbeiten
import torch 

# Ein Vektor mit 3 Einträgen 
# Das sind x-, y- und z-Koordinaten im Raum
x = torch.tensor([4, 6, 1])

print("Vektor x:")
print(x)

# Zugriff auf den zweiten Eintrag (Index 1, da bei 0 beginnend)
print("Zweiter Eintrag:", x[1])
```

Wie aussieht so ein Vektor im Raum?

```{code-cell} ipython3
# Wir importieren matplotlib für das Zeichnen und insbesondere das 3D-Plot-Modul
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# Wir erstellen ein neues Fenster für den Plot
fig = plt.figure()
# Wir fügen dem Plot eine 3D-Achse hinzu
ax = fig.add_subplot(projection="3d")

# Wir zeichnen den Vektor als Pfeil (quiver), der bei (0,0,0) startet
# und in Richtung (x[0], x[1], x[2]) zeigt
origin = [0,0,0]
ax.quiver(
    *origin,        # Startpunkt des Pfeils (Ursprung)
    *x,             # Zielrichtung des Pfeils (Vektorkomponenten)
    color="blue",   # Farbe des Pfeils
    arrow_length_ratio=0.1  # Verhältnis der Pfeilspitze zur Länge
    )

# Achesenbeschriftungen und Bereich
ax.set_xlim([-5,5])
ax.set_ylim([-5,5])
ax.set_zlim([-5,5])

# Wir beschriften die Achsen mit Klartext
ax.set_xlabel('X-Achse')
ax.set_ylabel('Y-Achse')
ax.set_zlabel('Z-Achse')

# Wir geben dem Plot einen Titel
ax.set_title('3D-Vektorvisualisierung')

# Wir passen das Layout an, damit nichts abgeschnitten wird
plt.tight_layout()

# Wir zeigen den Plot an
plt.show()
```

### 1.2. Matrix

Wir bezeichnen mit $A \in \mathbb{R}^{m \times n}$ eine **Matrix** mit $m$ Zeilen und $n$ Spalten, wobei $A_{i,j} \in \mathbb{R}$ der Eintrag in der $i$-ten Zeile und $j$-ten Spalte ist:

$$
A = \begin{pmatrix}
A_{1,1} & \dots & A_{1,n} \\
\vdots & & \vdots \\
A_{m,1} & \dots & A_{m,n}
\end{pmatrix}
\in \mathbb{R}^{m \times n}
$$


#### Erklärung

Eine **Matrix** ist wie eine Tabelle oder ein Rechteck aus Zahlen. Stell dir vor, du hast eine Tabelle mit mehreren **Zeilen** (von oben nach unten) und mehreren **Spalten** (von links nach rechts).

* $m$ = Anzahl der Zeilen (horizontal – wie bei einer Liste von Personen)
* $n$ = Anzahl der Spalten (vertikal – wie verschiedene Eigenschaften: z. B. Alter, Größe, Gewicht)

Jeder **Eintrag** in der Tabelle wird mit zwei Zahlen benannt:

* Die erste Zahl sagt, **in welcher Zeile** der Eintrag steht.
* Die zweite Zahl sagt, **in welcher Spalte** er steht.

Beispiel:
Der Eintrag ganz oben links heißt $A_{1,1}$.
Der Eintrag ganz unten rechts heißt $A_{m,n}$.


#### Beispiel
Stell dir eine Tabelle für drei Schüler vor, in der du ihre Mathe- und Deutschnoten einträgst:

|               | Mathe | Deutsch |
| ------------- | ----- | ------- |
| **Schüler 1** | 2     | 3       |
| **Schüler 2** | 1     | 2       |
| **Schüler 3** | 4     | 1       |

Das ist eine Matrix mit 3 Zeilen (Schüler) und 2 Spalten (Fächer).
So sieht sie als Matrix geschrieben aus:

$$
A = \begin{pmatrix}
2 & 3 \\
1 & 2 \\
4 & 1
\end{pmatrix}
$$

Eintrag in **2. Zeile und 1. Spalte** ist $A_{2,1} = 1$ → Das ist die Mathenote von Schüler 2.


#### 📌 Kurz gesagt:

> Eine Matrix ist eine Art **Zahlen-Tabelle**, mit der man in der Mathematik arbeitet – sie hat Zeilen und Spalten, und jeder Eintrag hat einen festen Platz.

---

+++

### 1.3. Einheitsmatrix

Die **Einheitsmatrix** $I \in \mathbb{R}^{m \times n}$ ist eine **quadratische Matrix**, bei der alle Einträge auf der **Diagonale** Eins sind und **alle anderen Null**:

$$
I = \begin{pmatrix}
1 & 0 & \dots & 0 \\
0 & \ddots & \ddots & \vdots \\
\vdots & \ddots & \ddots & 0 \\
0 & \dots & 0 & 1
\end{pmatrix}
$$

**Bemerkung:** Für alle Matrizen $A \in \mathbb{R}^{m \times n}$ gilt:
$A \times I = I \times A = A$


#### Erklärung

Die **Einheitsmatrix** ist wie eine besondere Tabelle mit Zahlen, bei der:

* **Nur auf der schrägen Linie von oben links nach unten rechts** stehen lauter Einsen.
* Alle anderen Felder in der Tabelle sind Null.
* Sie ist **quadratisch**, also hat **gleich viele Zeilen wie Spalten** (z. B. 2×2 oder 3×3).

Warum ist sie besonders?
Wenn du mit einer normalen Matrix rechnest und sie mit der Einheitsmatrix **multiplizierst**, **ändert sich nichts**!
Die Matrix bleibt gleich – wie beim Rechnen mit der Zahl 1:
→ Wenn du 7 × 1 rechnest, bleibt 7.

#### Beispiel

Stell dir vor, du hast eine Einheitsmatrix der Größe 3×3:

$$
I = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

Diese Matrix sagt sozusagen: **„Ich verändere nichts.“**

Wenn du eine andere Matrix hast, z. B.

$$
A = \begin{pmatrix}
5 & 2 & 7 \\
1 & 3 & 0 \\
4 & 6 & 8
\end{pmatrix}
$$

dann ist:

$$
A \times I = A \quad \text{und} \quad I \times A = A
$$


#### 📌 Kurz gesagt:

> Die Einheitsmatrix ist wie die Zahl „1“ für Matrizen: Wenn man mit ihr multipliziert, **bleibt alles gleich**.

---

+++

### 1.4. Diagonalmatrix

Eine **Diagonalmatrix** $D \in \mathbb{R}^{m \times n}$ ist eine **quadratische Matrix**, bei der **nur die Diagonale** (von oben links nach unten rechts) **nicht null ist** – alle anderen Einträge sind **null**:

$$
D = \begin{pmatrix}
d_1 & 0 & \dots & 0 \\
0 & \ddots & \ddots & \vdots \\
\vdots & \ddots & \ddots & 0 \\
0 & \dots & 0 & d_n
\end{pmatrix}
$$

Man schreibt auch einfach:

$$
D = \text{diag}(d_1, \dots, d_n)
$$


#### Erklärung

Eine **Diagonalmatrix** ist wie eine Tabelle, bei der nur die Zahlen auf der **schrägen Linie von links oben nach rechts unten** einen Wert haben – alle anderen Felder sind **Null**.

Anders als bei der Einheitsmatrix müssen die Zahlen **auf der Diagonale nicht Eins sein** – sie können **beliebige Zahlen** sein, z. B. 3, -2 oder 0.5.

#### Beispiel

Stell dir eine Diagonalmatrix mit 3 Zeilen und 3 Spalten vor:

$$
D = \begin{pmatrix}
2 & 0 & 0 \\
0 & -1 & 0 \\
0 & 0 & 5
\end{pmatrix}
$$

Oder kürzer geschrieben:

$$
D = \text{diag}(2, -1, 5)
$$

Das heißt:

* In der **1. Zeile, 1. Spalte** steht 2
* In der **2. Zeile, 2. Spalte** steht -1
* In der **3. Zeile, 3. Spalte** steht 5
* Alle anderen Stellen sind **0**


#### 📌 Kurz gesagt:

> Eine Diagonalmatrix ist eine Tabelle, bei der **nur die Diagonale zählt**.
> Sie ist wie eine Einheitsmatrix – aber statt Einsen stehen dort **beliebige Zahlen**.

---

+++

## 2. Matrixoperationen

+++

### 2.1. Vektor-Vektor-Multiplikation
#### 2.1.1. Definition

Es gibt zwei Arten von Produkten zwischen zwei Vektoren:

**• Skalarprodukt (inneres Produkt):**

Für $x, y \in \mathbb{R}^n$ gilt:

$$
x^T y = \sum_{i=1}^n x_i y_i \in \mathbb{R}
$$

**• äußeres Produkt (outer product):**

Für $x \in \mathbb{R}^m$, $y \in \mathbb{R}^n$ gilt:

$$
xy^T = \begin{pmatrix}
x_1 y_1 & \dots & x_1 y_n \\
\vdots & & \vdots \\
x_m y_1 & \dots & x_m y_n
\end{pmatrix}
\in \mathbb{R}^{m \times n}
$$

---

#### 2.1.2. Erklärung
Wenn man zwei Vektoren miteinander **multipliziert**, kann man das auf zwei verschiedene Arten tun:

🔹 Inneres Produkt (Skalarprodukt)

Das ist wie das **Zusammenzählen der passenden Paare** aus zwei Listen.

* Du nimmst den **ersten Wert von x** und **ersten Wert von y**, multiplizierst sie.
* Dann machst du das mit dem zweiten Paar, dritten Paar usw.
* Am Ende **addierst** du alle diese Produkte zusammen.

Das Ergebnis ist **eine einzelne Zahl** (ein Skalar).

**Beispiel:**
x = \[2, 3], y = \[4, 5]
→ Inneres Produkt: $2×4 + 3×5 = 8 + 15 = 23$


🔹 Äußeres Produkt

Hier baust du aus zwei Vektoren eine **ganze Tabelle** (Matrix).

* Du nimmst **jede Zahl aus x** und **multiplizierst sie mit jeder Zahl aus y**.
* Daraus entsteht eine Tabelle mit $m$ Zeilen (aus x) und $n$ Spalten (aus y).

**Beispiel:**
x = \[2, 3] (2 Werte → 2 Zeilen), y = \[4, 5, 6] (3 Werte → 3 Spalten)

→ Äußeres Produkt:

$$
xy^T = \begin{pmatrix}
2×4 & 2×5 & 2×6 \\
3×4 & 3×5 & 3×6
\end{pmatrix}
=
\begin{pmatrix}
8 & 10 & 12 \\
12 & 15 & 18
\end{pmatrix}
$$

---


#### 2.1.3 Anwendung im echten Leben

##### **Inneres Produkt (Skalarprodukt)**

**📍 Wann man es verwendet:**
Wenn man wissen will, **wie ähnlich oder „ausgerichtet“** zwei Dinge sind.

🧠 Typische Anwendungen:

1. **Ähnlichkeit zwischen zwei Objekten**
   – In der **Suche oder Empfehlungssystemen**:
   Zwei Vektoren (z. B. ein Nutzerprofil und ein Produktprofil) → Inneres Produkt misst, **wie gut sie zusammenpassen**.

2. **Physik: Arbeit = Kraft × Weg**
   – Wenn eine Kraft in eine bestimmte Richtung wirkt und sich ein Objekt in dieselbe Richtung bewegt.
   Das innere Produkt gibt an, **wie viel Arbeit wirklich geleistet wurde**.

3. **Maschinelles Lernen**
   – In linearen Modellen (z. B. bei der Vorhersage):
   Das Ergebnis eines Modells ist oft ein **inneres Produkt zwischen den Gewichten und den Eingabewerten**.

##### **Äußeres Produkt**

**📍 Wann man es verwendet:**
Wenn man aus zwei Dingen **eine Beziehungsmatrix oder Struktur** aufbauen will.

🧠 Typische Anwendungen:

1. **Korrelation von Datenstrukturen**
   – Wenn du zwei Datenvektoren hast und wissen willst, **wie sie sich gegenseitig beeinflussen** – z. B. bei Covarianzmatrizen.

2. **Bildverarbeitung oder neuronale Netze**
   – Wenn man **Beziehungen zwischen Merkmalen** lernen will.

3. **Erzeugung von Matrizen aus Basisvektoren**
   – In der **Linearen Algebra**, wenn man Matrizen aus einfachen Richtungsvektoren zusammenbauen will.

---

#### 2.1.4. Beispiel 
##### Inneres Produkt : Vorhersage mit einem linearen Modell

Wir wollen entscheiden, ob eine Person ein Produkt kaufen wird, basierend auf diesen **Merkmalen**:

| Merkmal                              | Eingabewert $x_i$ | Gewicht $w_i$ | Beitrag $x_i \cdot w_i$ |
| ------------------------------------ | ----------------- | ------------- | ----------------------- |
| Anzahl Seitenbesuche                 | 4                 | 0.3           | 1.2                     |
| Gekaufte ähnliche Produkte           | 2                 | 0.8           | 1.6                     |
| Verweildauer (in Minuten)            | 10                | 0.1           | 1.0                     |
| **Gesamtergebnis (inneres Produkt)** |                   |               | **3.8**                 |


🟨 Interpretation

* **Eingabewerte**: Was wir über die Person wissen
* **Gewichte**: Wie wichtig jedes dieser Merkmale ist
* **Beitrag**: Das Produkt aus Wert × Gewicht (also das, was am Ende zählt)

Der **Score = 3.8** ergibt sich aus:

$$
x^T w = 4 \cdot 0.3 + 2 \cdot 0.8 + 10 \cdot 0.1 = 1.2 + 1.6 + 1.0 = 3.8
$$


🧸 Alltagsbild

Das ist wie eine **Bewertung mit Punkten**:

* Besuch = 0.3 Punkte pro Mal
* Kauf = 0.8 Punkte pro Produkt
* Verweildauer = 0.1 Punkte pro Minute

Dann rechnest du einfach alle Punkte zusammen, um zu entscheiden, **wie interessiert** die Person ist.

📌 Merksatz:

> Das **innere Produkt** ist eine elegante Art zu sagen:
> „**Wie viele Punkte bekommt jemand insgesamt?**“
> – durch das Multiplizieren der Eingaben mit ihrer Bedeutung (Gewicht).


##### Äußeres Produkt: Bewertung von Filmen durch Personen

🎭 Personen:

Du hast **2 Personen** – nennen wir sie Anna und Ben.
Jede Person hat eine **„Vorliebe“ für Genres** in Form eines Vektors:

$$
x = \begin{pmatrix} 0.9 \\ 0.4 \end{pmatrix}
$$

* $x_1 = 0.9$: Anna liebt Action
* $x_2 = 0.4$: Ben mag Action etwas, aber weniger

🎬 Filme:

Du hast **3 Filme**, jeder mit einem **Genre-Gewicht** (wie „Wieviel Action ist drin“):

$$
y = \begin{pmatrix} 0.6 \\ 0.3 \\ 0.9 \end{pmatrix}
$$

* Film 1: 60 % Action
* Film 2: 30 % Action
* Film 3: 90 % Action


🔹 Schritt: Äußeres Produkt berechnen

Du rechnest:

$$
xy^T = \begin{pmatrix}
0.9 \times 0.6 & 0.9 \times 0.3 & 0.9 \times 0.9 \\
0.4 \times 0.6 & 0.4 \times 0.3 & 0.4 \times 0.9
\end{pmatrix}
=
\begin{pmatrix}
0.54 & 0.27 & 0.81 \\
0.24 & 0.12 & 0.36
\end{pmatrix}
$$


🟨 Wie liest man das?

* Die **Zeilen** stehen für Personen:
  Zeile 1 = Anna, Zeile 2 = Ben

* Die **Spalten** stehen für Filme:
  Spalte 1 = Film 1, Spalte 2 = Film 2, Spalte 3 = Film 3

🧾 Ergebnis-Matrix:

|          | Film 1 | Film 2 | Film 3 |
| -------- | ------ | ------ | ------ |
| **Anna** | 0.54   | 0.27   | 0.81   |
| **Ben**  | 0.24   | 0.12   | 0.36   |



✅ Bedeutung:

* Diese Werte zeigen eine Art **„Interessenslevel“** oder **„Kompatibilität“** zwischen Person und Film.
* Anna ist **sehr interessiert** an Film 3 (0.81), weniger an Film 2 (0.27).
* Ben interessiert sich **mäßig** für Film 1 (0.24) und **kaum** für Film 2 (0.12).



📌 Fazit:

> Mit dem **äußeren Produkt** kannst du aus zwei Vektoren – einem für Personen und einem für Filme – eine **Bewertungsmatrix** erzeugen, die alle möglichen Kombinationen abbildet.
> Das ist nützlich in **Empfehlungssystemen**, z. B. bei Netflix oder Spotify.

#### 2.1.5. Kurz zusammengefasst:

| Produktart      | Typisches Ergebnis | Was es bedeutet                           | Anwendung im echten Leben                   |
| --------------- | ------------------ | ----------------------------------------- | ------------------------------------------- |
| Inneres Produkt | Zahl               | „Wie gut passen zwei Dinge zusammen?“     | Empfehlungen, Vorhersagen, Arbeit in Physik |
| Äußeres Produkt | Matrix             | „Wie hängen alle Kombinationen zusammen?“ | Beziehungen, Datenstruktur, neuronale Netze |

---

+++

### 🟦 Übersetzung ins Deutsche

#### 2.2. Matrix-Vektor-Multiplikation

##### 2.2.1. Definition

Das Produkt einer Matrix $A \in \mathbb{R}^{m \times n}$ mit einem Vektor $x \in \mathbb{R}^n$ ergibt einen **Vektor der Länge $m$**:

$$
Ax = \begin{pmatrix}
a^T_{r,1}x \\
\vdots \\
a^T_{r,m}x
\end{pmatrix}
= \sum_{i=1}^n a_{c,i} x_i \in \mathbb{R}^m
$$

Dabei ist:

* $a^T_{r,i}$: die **i-te Zeile** von A (als Vektor)
* $a^T_{c,j}$: die **j-te Spalte** von A (als Vektor)
* $x_i$: der **i-te Eintrag** des Vektors $x$

---

### 🟨 Erklärung in einfachem Deutsch (ohne mathematische Symbole)

Wenn du eine **Matrix mit einem Vektor multiplizierst**, passiert Folgendes:

1. **Jede Zeile der Matrix** wird mit dem Vektor **einzeln verrechnet**.
2. Das heißt: Du berechnest für jede Zeile ein **inneres Produkt** (siehe oben).
3. Am Ende bekommst du **einen neuen Vektor**, der so viele Einträge hat wie die Matrix **Zeilen**.

---

### 🧸 Beispiel mit Tabelle (konkret)

Nehmen wir eine Matrix $A$ mit **2 Zeilen und 3 Spalten** (also 2×3):

|             | Merkmal 1 | Merkmal 2 | Merkmal 3 |
| ----------- | --------- | --------- | --------- |
| **Zeile 1** | 1         | 2         | 3         |
| **Zeile 2** | 4         | 5         | 6         |

Und einen Vektor $x$ mit 3 Einträgen:

$$
x = \begin{pmatrix} 1 \\ 0 \\ 2 \end{pmatrix}
$$

#### 📏 Rechnen:

* Erste Zeile × Vektor:
  $1×1 + 2×0 + 3×2 = 1 + 0 + 6 = 7$

* Zweite Zeile × Vektor:
  $4×1 + 5×0 + 6×2 = 4 + 0 + 12 = 16$

#### 🟩 Ergebnis (neuer Vektor):

$$
Ax = \begin{pmatrix} 7 \\ 16 \end{pmatrix}
$$

---

### 📌 Was bedeutet das?

* Die Matrix $A$ enthält **z. B. Informationen über zwei Produkte**.
* Der Vektor $x$ enthält **z. B. deine Vorlieben** für drei Eigenschaften (z. B. Preis, Design, Leistung).
* Das Produkt $Ax$ ergibt einen neuen Vektor – **eine Bewertung pro Produkt**, basierend auf deinen Vorlieben.

---

### 🧠 Merksatz:

> Matrix-Vektor-Multiplikation = „**mehrere gewichtete Punktesummen auf einmal rechnen**“ – eine pro Zeile der Matrix.

+++
