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

Simulátor má konfigurační soubor **[`config.json`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/config.json)** (kompletní manuál k parametrům viz **[`docs/configuration.md`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/docs/configuration.md)**) a hlavní spouštěcí skript **[`main.py`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/main.py)** (nebo konzolový příkaz `dust-impact`).

### 1. Spuštění z příkazové řádky (CLI):

```bash
# Základní spuštění podle parametrů v config.json:
python main.py

# Běh na serveru s přesměrováním všech výstupů do zadané složky:
python main.py --config config.json --output-dir /cesta/k/vysledkum --no-visualize

# Rychlý post-processing ze surových dat (přeskočí PIC výpočet a jen vygeneruje grafy):
python main.py --config config.json --output-dir /cesta/k/vysledkum --plot-only

# Běžné spuštění podle jiné konfigurace (např. šablona):
python main.py --config inputs/config_template.json

# Uložení do formátu NPZ místo výchozího HDF5:
python main.py --format npz

# Export pro ParaView (.vti pole, .vtp částice a .pvd časová osa):
python main.py --export-vtk

# Spuštění přes instalovaný příkaz:
dust-impact --config inputs/config_template.json --output-dir outputs/run_01 --format h5 --export-vtk
```

### 2. Spuštění z Python skriptu:

```python
from dust_impact import main

# Spustí výpočet a vizualizace podle config.json:
main(
    config_file="config.json",
    output_dir="outputs/run_01",
    plot_only=False,
    visualize_results=False,
    output_format="h5",
    export_vtk=True,
    enable_checkpointing=True
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
| `--output-dir <složka>` | `-o` | Cílový adresář pro veškeré výstupy (`.h5`, `.npz`, `.csv`, `.png`, `.gif`, VTK). Pokud neexistuje, automaticky se vytvoří. |
| `--format <h5\|npz>` | | Formát ukládání dat simulace (výchozí: `h5` s hierarchickou strukturou a gzip kompresí; volitelně `npz` se self-contained JSON metadaty). |
| `--export-vtk` | | **ParaView export:** Vygeneruje 3D `.vti` mřížky potenciálu a hustoty, `.vtp` mračna částic a master kolekce `.pvd` pro přímou vizualizaci v ParaView. |
| `--no-checkpoint` | | Vypne periodické ukládání rotujících kontrolních bodů (`.checkpoint.h5` / `.checkpoint.npz`). |
| `--plot-only` | | **Režim post-processingu:** Přeskočí fyzikální simulaci, načte existující data (`.h5` nebo `.npz`) a vygeneruje grafy/animace či VTK export. |
| `--no-visualize` | | Vypne interaktivní okna grafů (vynutí headless vykreslování přímo do souborů na disku). |
| `--visualize` | | Vynutí interaktivní okna grafů (`plt.show()`), vyžaduje GUI/X11. |

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

## 🏗️ Architektura a samostatné moduly projektu

Projekt je navržen podle principů modulární vědecké architektury s jasným oddělením odpovědností:

```
dust_impact/
├── geometry/         # Samostatný modul pro geometrii, voxelizaci a přípravu vstupů
│   ├── voxelizer.py  # Detekce vodivých těles, prahování gradientu, binární masky (koule, kvádr, válec)
│   ├── surface.py    # Ray-tracing na síti i voxelové mřížce, normála povrchu, bod dopadu prachu
│   ├── analytical.py # Analytická tělesa (koule, box) s přesným potenciálem V a polem E (Laplace / Debye)
│   ├── spis_loader.py# Načítání SPIS VTK polygonálních sítí, ImageData vzorkování, enclosed-points testy
│   └── prepared.py   # Kontejner PreparedGeometry3D a orchestrátor build_simulation_geometry
├── physics/          # Analytické a empirické fyzikální zákony
│   ├── ramo_shockley.py # Výpočet indukovaných proudů do antén v 3D
│   ├── charging.py   # Nabíjení těles v plazmatu (Cassini / Solar Orbiter OML teorie)
│   └── constants.py  # Fyzikální konstanty (SI jednotky)
├── numerics/         # Numerické algoritmy a jádra řešičů
│   ├── poisson.py    # 3D Poissonův řešič (PyAMG algebraický multigrid / SciPy sparse BiCGSTAB)
│   ├── pushers.py    # 3D Boris / Leap-Frog částicový integrátor a CFL stabilita
│   └── interpolators.py # Trilineární 3D interaktivní interpolátory (Cloud-in-Cell)
├── common/           # Společná infrastruktura
│   ├── io.py         # Ukládání a načítání HDF5 (.h5), NPZ s JSON metadaty a rotující checkpointy
│   ├── vtk_export.py # Export časových řad polí a částic pro ParaView (.vti, .vtp, .pvd)
│   └── circuits.py   # Integrování RC odezvy anténního předzesilovače
└── sim3d/            # Řídicí vrstva 3D PIC simulace
    ├── ensemble.py   # Správa částicového ansámblu (ParticleEnsemble), injekce a CCD kolize
    ├── field_solver.py # Řešič Poissonovy rovnice a elektrostatických polí (FieldSolver3D)
    ├── collector.py  # Sběr náboje a integrace anténních RC obvodů (AntennaCircuitCollector)
    ├── sim_core.py   # Hlavní orchestrátor 3D simulace (DustImpactSimulation3D)
    ├── params.py     # Dataclass modely parametrů simulace (SimulationParams3D)
    ├── config_loader.py # Načítání JSON konfigurace, validace a normalizace cest
    ├── runner.py     # CLI orchestrátor běhu simulace
    └── plotting.py   # Vykreslování grafů a generování animací

```

---

## 🧪 Spuštění verifikačních testů

Projekt disponuje ucelenou testovací sadou **53 unit testů** pokrývajících:
1. **Numeriku a jádra (`test_numerics.py`):**
   * Cloud-in-Cell (CIC) trilineární vážení náboje s exaktním zachováním náboje $\sum \rho \cdot dV = \sum q$.
   * Adjointní trilineární interpolace elektrických polí $\mathbf{E} \to \mathbf{x}_p$.
   * Symplektický Boris/Leap-frog posun částic a CFL stabilita.
2. **Zpracování geometrie (`test_geometry.py`):**
   * Voxelizace geometrických těles (koule, kvádry, válce).
   * Detekce kovových povrchů z gradientu váhových potenciálů (`detect_metal_mask_3d`).
   * Analytické elektrostatické modely (vakuum i Debyeovo stínění, shoda s analytickými vztahy $V(r)$ a $\mathbf{E}(r)$).
   * Ray-tracing na mřížce a výpočet výchozího bodu a normály dopadu.
   * Kontejner `PreparedGeometry3D` a integrace do řešiče `DustImpactSimulation3D`.
3. **Analytickou 3D fyzikální verifikaci (`test_physics_analytical_3d.py`):**
   * 3D Ramo-Shockley odezva antény na letící náboj (přesný průběh $I(t)$ a píky proudu).
   * 3D Poissonův řešič pro Gaussovský nábojový oblak ($V(r) \sim \frac{\text{erf}(r/\sigma)}{r}$).
   * Metoda vytvořených řešení (MMS) a ověření 2. řádu konvergence $\mathcal{O}(dx^2)$ diferenčního operátoru.
   * 3D plazmatické Langmuirovy oscilace (FFT spektrum kmitů částic vs. $\omega_{pe}$).
4. **Fyzikální zákony a obvody (`test_physics_level1.py`, `test_physics_level2.py`):**
   * Globální zachování náboje v 3D PIC, Debyeovo stínění, analytická odezva RC obvodu antény, ambipolární expanze.
5. **I/O formáty a ParaView export (`test_io_formats.py`):**
   * HDF5, NPZ s JSON metadaty, rotující checkpointy a generování `.vti`/`.vtp`/`.pvd`.

Spuštění všech 53 testů:
```bash
python -m unittest discover -s tests
```

*Výstupní protokoly, porovnávací grafy a animace z testování se automaticky ukládají do složky `outputs/tests/`.*

---

## 📂 Výstupní data

Všechny vygenerované artefakty simulací se automaticky ukládají do složky **[`outputs/`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/outputs)** (nebo do složky zadané parametrem `--output-dir`):
* Datové archivy **`.h5`** (primární HDF5 se strukturou signálů, polí a částic) a **`.npz`** (alternativní formát se zabudovanými JSON metadaty)
* Kontrolní body **`.checkpoint.h5`** / **`.checkpoint.npz`** (automatická ochrana výpočtu proti pádu systému)
* ParaView 3D scény **`paraview_vtk/`** (`fields.pvd`, `*.vti`, `particles_*.vtp`)
* Výstupní animace **`.gif`**
* Statické grafy **`.png`**
* Exportované časové řady **`.csv`**