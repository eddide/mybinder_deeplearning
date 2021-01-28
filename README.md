# Binder Badge
https://mybinder.org/v2/gh/eddide/mybinder_logreg/HEAD

# Dokumentation zur Ausführung
1. Aktivieren des Links zu Binder
2. Öffnen des Python-Notebooks "Installationen"
3. In der Toolbar unter "Kernel" "Restart & Run All" aktivieren. --> Bestätigen des Pop-up Fensters mit "Restart and Run All Cells"
4. Nach der erfolgreichen Ausführung Schließen des Python-Notenooks "Installationen"
5. Öffnen des Pyhton-Notebooks "Logistic Regression"
6. In der Toolbar unter "Kernel" "Restart & Run All" aktivieren. --> Bestätigen des Pop-up Fensters mit "Restart and Run All Cells"
7. Ergebnisse mit den unten beschriebenen Ergebnissen abgleichen.


# Klassifikation mit Keras TF 2.0 - Programmierprojekt

Lasst uns ein Klassifikationsproblem mit der Keras API für TF 2.0 bearbeiten!

## Der Datensatz

### Brustkrebs in Wisconsion (diagnostisch)
--------------------------------------------

**Datensatz Eigenschaften:**

    :Anzahl Einträge: 569

    :Anzahl Attribute: 30 numerische, prädiktive Attribute und die Klasse

    :Attributinformation:
        - Radius (Durchschnitt der Distanzen des Zentrums zum Perimeter)
        - Textur (Standardabweichung von Graustufen)
        - Perimeter
        - Fläche
        - Glattheit (lokale Variation der Radiuslänge)
        - Kompaktheit (Perimeter^2 / Fläche - 1.0)
        - Konkavität (Gewichtigkeit konkaver Anteile an der Kontur)
        - Konkave Punkte (Anzahl konkaver Punkte in der Kontur)
        - Symmetrie 
        - fraktale Dimension ("coastline approximation" - 1)

        Der Durchschnitt, die Standardabweichung und "schlimmster" oder 
        größte (Durchschnitt der drei größten Werte) dieser Features wurde 
        berechnet für jedes Bild, was 30 Features ergibt. Beispielsweise ist
        Feld 3 durchschnittlicher Radius, Feld 13 Radius SE, Feld 23 
        schlimmster Radius.

        - Klassen:
                - WDBC-Malignant
                - WDBC-Benign

    :Zusammengefasste Statistiken:

    ===================================== ====== ======
                                           Min    Max
    ===================================== ====== ======
    Radius (Durchschnitt):                 6.981  28.11
    Textur (Durchschnitt):                 9.71   39.28
    Perimeter (Durchschnitt):              43.79  188.5
    Fläche (Durchschnitt):                 143.5  2501.0
    Glattheit (Durchschnitt):              0.053  0.163
    Kompaktheit (Durchschnitt):            0.019  0.345
    Konkavität (Durchschnitt):             0.0    0.427
    Konkave Punkte (Durchschnitt):         0.0    0.201
    Symmetrie (Durchschnitt):              0.106  0.304
    fraktale Dimension (Durchschnitt):     0.05   0.097
    Radius (Standardabw.):                 0.112  2.873
    Textur (Standardabw.):                 0.36   4.885
    Perimeter (Standardabw.):              0.757  21.98
    Fläche (Standardabw.):                 6.802  542.2
    Glattheit (Standardabw.):              0.002  0.031
    Kompaktheit (Standardabw.):            0.002  0.135
    Konkavität (Standardabw.):             0.0    0.396
    Konkave Punkte (Standardabw.):         0.0    0.053
    Symmetrie (Standardabw.):              0.008  0.079
    fraktale Dimension (Standardabw.):     0.001  0.03
    Radius (schlimmster):                  7.93   36.04
    Textur (schlimmster):                  12.02  49.54
    Perimeter (schlimmster):               50.41  251.2
    Fläche (schlimmster):                  185.2  4254.0
    Glattheit (schlimmster):               0.071  0.223
    Kompaktheit (schlimmster):             0.027  1.058
    Konkavität (schlimmster):              0.0    1.252
    Konkave Punkte (schlimmster):          0.0    0.291
    Symmetrie (schlimmster):               0.156  0.664
    fraktale Dimension (schlimmster):      0.055  0.208
    ===================================== ====== ======

    :Fehlende Attributwerte: None

    :Klassenverteilung: 212 - Malignant, 357 - Benign

    :Erstellt von:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian

    :Spender: Nick Street

    :Datum: November, 1995

Dies ist eine Kopie des UCI ML Breast Cancer Wisconsin (Diagnostic) Datensatzes.
https://goo.gl/U2Uwz2

Features wurden berechnet mittels digitalisierter Bilder von fine needle
aspirate (FNA)-Aufzeichnungen von Brüsten. Sie beschreiben Charakteristiken
der im Bild enthaltenen, zellulären  Nuklei.

Die obig beschriebene Trennebenewurde mit der Multisurface 
Method-Tree (MSM-T) ermittelt [K. P. Bennett, "Decision Tree
Construction Via Linear Programming." Proceedings of the 4th
Midwest Artificial Intelligence and Cognitive Science Society,
pp. 97-101, 1992], einer auf linearer Programmierung basierenden
Klassifizierungsmethode zum Aufbau eines Entscheidungsbaums.
Relevante Features wurden mit einer erschöpfenden Suche über
1-4 Features und 1-3 Trennebenen errechnet.

Das zur Berechnung der Trennebenen im dreidimensionalen Raum
verwendete lineare Programm ist beschrieben in:
[K. P. Bennett and O. L. Mangasarian: "Robust Linear
Programming Discrimination of Two Linearly Inseparable Sets",
Optimization Methods and Software 1, 1992, 23-34].

Diese Datenbank ist auch erhältlich auf dem UW CS ftp server:

ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

.. : Referenzen

   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
     San Jose, CA, 1993.
   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
     prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
     July-August 1995.
   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
     163-171.

## Anwendung
### Libraries Import
zuerst werden die relevanten Libraries importiert
- pandas
- numpy
- matplotlib
- seaborn
### Daten Import und Überprüfung
dann werden die Daten importiert und überprüft
- Anzeigen der Daten
- Information der Daten
- Beschreibung der Daten
### Explorative Datenanalyse
Verwendung unterschiedlicher Visualisierungen um die Daten zu erforschen
- Histogramm
- Jointplot
- Pairplot
### Logistische Regression
Anwenden der logistischen Regression zur Vorhersage der Kategorie
- Aufsplitten in Trainings- und Testdaten
- Trainieren und Fitten des Modells auf den Datensatz
### Evaluation der Vorhersage
Evaluierung über den classification report.
Es sollten ähnliche Ergebnisse wie diese angezeigt werden:
- accuracy: 91%
- reall: 96 und 85%
- f1-score: 91 und 90 %
