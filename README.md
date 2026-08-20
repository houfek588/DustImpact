# 🚀 DustImpact (PostProcessSPIS)

**DustImpact** je vědecký fyzikálně-numerický simulátor dopadů mikrometeoroidů na trupy vesmírných sond (např. Solar Orbiter) v plazmatu slunečního větru. Simuluje kinetiku expanze impaktního plazmatického mraku (PIC metoda) a indukovanou i dopadovou proudovou odezvu anténních systémů.

---

## 🛠️ Instalace a požadavky

Simulátor vyžaduje Python 3.8+ a standardní vědecké knihovny:

```bash
# Klonování a instalace balíčku v editačním režimu:
pip install -e .

# Instalace s volitelnou podporou 3D VTK sítí (PyVista):
pip install -e .[3d]
```

---

## 🚀 Příkazy pro spuštění simulátoru

Simulátor má jediný zastřešující konfigurační soubor **[`config.json`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/config.json)** a hlavní spouštěcí skript **[`main.py`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/main.py)**.

### 1. Spuštění z příkazové řádky (CLI):

```bash
# Základní spuštění podle parametru "dim" v config.json:
python main.py

# Vynucení 2D výpočtu:
python main.py --dim 2

# Vynucení 3D výpočtu:
python main.py --dim 3

# Vypnutí grafické vizualizace (vhodné pro výpočty na pozadí či HPC):
python main.py --no-visualize

# Vynucení zapnutí vizualizace:
python main.py --visualize

# Zadání vlastního konfiguračního souboru:
python main.py --config inputs/config.json --dim 3 --no-visualize
```

### 2. Spuštění z Python skriptu:

```python
from dust_impact import main

# Spustí výpočet a vizualizace podle config.json:
main(config_file="config.json", dim_override=2, visualize_results=True)
```

---

## 🧪 Spuštění verifikačních testů

Všechny jednotkové i fyzikální testy (zachování náboje, Ramo-Shockleyho teorém, RC obvody a ambipolární kinetika) se spouští příkazem:

```bash
python -m unittest discover -s tests
```

*Výstupní protokoly a animace z testování se ukladají do složky `outputs/tests/`.*

---

## 📂 Výstupní data

Všechny vygenerované artefakty simulací se automaticky ukládají do složky **[`outputs/`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/outputs)**:
* Datové binary archivy `.npz`
* Výstupní animace `.gif`
* Statické grafy `.png`
* Exportované časové řady `.csv`

Bylo by možné zadat oblak plazmatu různými způsoby? Tedy tím, jak je to teď (oblak plazmatu koncentrovaný v jednom místě) nebo druhou možností homogenně
  rozmístěnými částicemi plazmatu.