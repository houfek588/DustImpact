# 🚀 DustImpact (PostProcessSPIS)

**DustImpact** je vědecký fyzikálně-numerický simulátor dopadů mikrometeoroidů na trupy vesmírných sond (např. Solar Orbiter) v plazmatu slunečního větru. Simuluje kinetiku expanze impaktního plazmatického mraku (PIC metoda) a indukovanou i dopadovou proudovou odezvu anténních systémů.

---

## 🛠️ Instalace a požadavky

Simulátor vyžaduje Python 3.8+ a standardní vědecké knihovny:

```bash
# Instalace přes requirements.txt (vhodné pro servery a virtuální prostředí):
pip install -r requirements.txt

# Nebo instalace balíčku v editačním režimu (zpřístupní i systémový příkaz 'dust-impact'):
pip install -e .

# Instalace s volitelnou podporou 3D VTK sítí (PyVista):
pip install -e .[3d]
```

---

## 🚀 Příkazy pro spuštění simulátoru

Simulátor má konfigurační soubor **[`config.json`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/config.json)** a hlavní spouštěcí skript **[`main.py`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/main.py)** (nebo konzolový příkaz `dust-impact`).

### 1. Spuštění z příkazové řádky (CLI):

```bash
# Základní spuštění podle parametrů v config.json:
python main.py

# Běh na serveru s přesměrováním všech výstupů do zadané složky:
python main.py --config config.json --output-dir /cesta/k/vysledkum --no-visualize

# Rychlý post-processing ze surových dat (přeskočí PIC výpočet a jen vygeneruje grafy):
python main.py --config config.json --output-dir /cesta/k/vysledkum --plot-only

# Vynucení 2D / 3D výpočtu:
python main.py --dim 2
python main.py --dim 3

# Spuštění přes instalovaný příkaz:
dust-impact --config inputs/config_3d_template.json --output-dir outputs/run_01
```

### 2. Spuštění z Python skriptu:

```python
from dust_impact import main

# Spustí výpočet a vizualizace podle config.json:
main(
    config_file="config.json",
    output_dir="outputs/run_01",
    plot_only=False,
    dim_override=3,
    visualize_results=False
)
```

---

## 🖥️ Návod na zprovoznění a spouštění na serveru

### 1. Příprava prostředí na vzdáleném serveru (Linux / HPC)

Po přenesení projektu na server (např. přes `git clone`, `rsync` nebo `scp`) doporučujeme vytvořit izolované virtuální prostředí:

```bash
# 1. Přejděte do složky projektu
cd /cesta/k/PostProcessSPIS

# 2. Vytvořte virtuální prostředí (venv)
python3 -m venv venv

# 3. Aktivujte prostředí
source venv/bin/activate

# 4. Aktualizujte pip a nainstalujte závislosti
pip install --upgrade pip
pip install -r requirements.txt

# 5. (Volitelné) Nainstalujte balíček pro globální příkaz 'dust-impact':
pip install -e .
```

---

### 2. Klíčové parametry CLI pro server

| Přepínač | Zkratka | Popis |
| :--- | :--- | :--- |
| `--config <cesta>` | `-c` | Cesta ke konfiguračnímu souboru JSON (výchozí: `config.json`). |
| `--output-dir <složka>` | `-o` | Cílový adresář pro veškeré výstupy (`.npz`, `.csv`, `.png`, `.gif`). Pokud neexistuje, automaticky se vytvoří. |
| `--plot-only` | | **Režim post-processingu:** Přeskočí fyzikální simulaci, načte existující `.npz` a pouze vygeneruje grafy a animace. |
| `--no-visualize` | | Vypne interaktivní okna grafů (vynutí headless vykreslování přímo do souborů na disku). |
| `--visualize` | | Vynutí interaktivní okna grafů (`plt.show()`), vyžaduje GUI/X11. |
| `--dim <2\|3>` | | Přepíše dimenzi simulace (2D nebo 3D) bez nutnosti měnit JSON. |

---

### 3. Příkladové scénáře spuštění na serveru

#### Scénář A: Kompletní výpočet a vygenerování grafů do složky úlohy
Skript načte VTK geometrie z `inputs/`, provede PIC simulaci a všechny výsledky i grafy uloží do dedikované složky:
```bash
python main.py --config config.json --output-dir /mnt/data/sim_run_01 --no-visualize
```

#### Scénář B: Rychlý post-processing ze surových dat (`--plot-only`)
Pokud již na serveru proběhla simulace a existuje soubor `/mnt/data/sim_run_01/out_vysledky.npz`, můžete okamžitě přegenerovat grafy bez nového časově náročného počítání:
```bash
python main.py --config config.json --output-dir /mnt/data/sim_run_01 --plot-only
```

#### Scénář C: Spuštění na pozadí přes `nohup` (nezávisle na SSH relaci)
```bash
nohup python main.py --config config.json --output-dir outputs/run_heavy --no-visualize > run.log 2>&1 &

# Průběžná kontrola výpisu simulace:
tail -f run.log
```

#### Scénář D: Dávkový skript pro plánovač úloh SLURM (`job.sh`)
```bash
#!/bin/bash
#SBATCH --job-name=dust_pic
#SBATCH --output=slurm_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

# Aktivace virtuálního prostředí
source /cesta/k/PostProcessSPIS/venv/bin/activate

# Spuštění simulace s headless exportem
dust-impact --config /cesta/k/config.json --output-dir /mnt/results/${SLURM_JOB_ID} --no-visualize
```

---

### 4. Automatické chování v headless režimu

* **Matplotlib backend:** Na Linuxových serverech bez grafického prostředí (bez `$DISPLAY`) nebo při zadání `--no-visualize` / `--plot-only` skript automaticky inicializuje neinteraktivní backend `matplotlib.use('Agg')`. Všechna volání `plt.show()` a manipulace s GUI okny jsou bezpečně přeskočena.
* **Vyhodnocování cest v JSONu:** Všechny relativní cesty k datům a VTK souborům v konfiguračním JSONu jsou automaticky vyhodnocovány vůči umístění daného JSON souboru. Skript tak lze spolehlivě volat z jakéhokoliv pracovního adresáře.

---

## 🧪 Spuštění verifikačních testů

Všechny jednotkové i fyzikální testy (zachování náboje, Ramo-Shockleyho teorém, RC obvody, ambipolární kinetika, CLI a přesměrování výstupů) se spouští příkazem:

```bash
python -m unittest discover -s tests
```

*Výstupní protokoly a animace z testování se ukládají do složky `outputs/tests/`.*

---

## 📂 Výstupní data

Všechny vygenerované artefakty simulací se automaticky ukládají do složky **[`outputs/`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/outputs)** (nebo do složky zadané parametrem `--output-dir`):
* Datové binární archivy `.npz`
* Výstupní animace `.gif`
* Statické grafy `.png`
* Exportované časové řady `.csv`