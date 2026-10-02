# 🚀 Architektura a logika 3D PIC Simulátoru (`dust_impact.sim3d`)

Tento dokument definuje systémovou architekturu, datový workflow a fyzikálně-numerická specifika 3D simulátoru hyperrychlých dopadů prachu v balíčku `dust_impact.sim3d`.

Tento simulátor byl navržen pro analýzu signálů na vesmírných sondách (např. Solar Orbiter, Parker Solar Probe, laboratorní experimenty), kde dopad prachové částice rychlostí desítek km/s generuje oblak hustého plazmatu. Expanze tohoto mraku a dynamika okolního plazmatu slunečního větru způsobuje komplexní elektromagnetické rušení, indukci napětí a sběr náboje na detekčních anténách.

---

## 1. High-Level Architektura (Návrhový vzor Přístup B)

Systém je striktně modulární a odděluje fyzikální výpočty (backend) od uživatelského rozhraní, I/O operací a vykreslování (frontend). Architektura fúzuje dva programovací přístupy:

- **Data-Oriented Design (DOD)**: Použitý v horkých smyčkách (hot loops) uvnitř výpočetního jádra (`DustImpactSimulation3D`) pro dosažení maximální propustnosti procesoru a optimálního využití L1/L2 cache (operace nad velkými souvislými bloky NumPy polí).
- **Object-Oriented Programming (OOP)**: Použitý pro vnější zapouzdření stavů, správu životního cyklu simulace a udržení čistého jmenného prostoru (namespaces).

### Základní moduly v `dust_impact.sim3d` a jejich role:

- **`dust_impact.sim3d.runner`** (`run_3d_simulation`): Řídí lineární workflow: načtení konfigurace $\rightarrow$ načtení VTK geometrií/superpozice $\rightarrow$ inicializace 3D solveru $\rightarrow$ běh simulace $\rightarrow$ uložení surových dat do `outputs/` $\rightarrow$ spuštění vizualizace.
- **`dust_impact.sim3d.params`** (Data Models): Dataclasses pro 3D parametry simulace (`SimulationParams3D`, `SimulationToggles3D`, `PlottingConfig3D`, `PhysicConfig`, `NumericConfig`, `ImpactConfig`) s dynamickými `@property` pro rozměry sítě a polohy.
- **`dust_impact.sim3d.config_loader`** (Config Layer): Robustní načítání a validace parametrů z JSON souborů (`setup_simulation_parameters_3d`), normalizace mřížek a adaptivní přesměrování výstupních cest.
- **`dust_impact.geometry.config`**: Dataclasses pro definici SPIS a analytických geometrií (`GeometryConfig`, `SpisGeometryConfig`, `AnalyticalGeometryConfig`, `VTKFilesConfig`).
- **`dust_impact.sim3d.ensemble`** (`ParticleEnsemble`): Správa fázového prostoru elektronů a iontů, Leapfrog integrace a Continuous Collision Detection (CCD).
- **`dust_impact.sim3d.field_solver`** (`FieldSolver3D`): Cloud-in-Cell (CIC) depozice náboje, Algebraic Multigrid (PyAMG) Poissonův řešič a výpočet zrychlení.
- **`dust_impact.sim3d.collector`** (`AntennaCircuitCollector`): Sledování indukovaných Ramo-Shockleyho proudů, sběr náboje a integrace anténních RC obvodů.
- **`dust_impact.sim3d.sim_core`** (Compute Layer): Srdce celého 3D systému (`DustImpactSimulation3D`). Samostatný objekt PIC (Particle-in-Cell) simulátoru nezávislý na GUI, orchestrátor subsystémů s metodou `step()`.
- **`dust_impact.sim3d.plotting`** (Presentation Layer): Prezentační vrstva (`plot_simulation_results_3d`). Generuje statické grafy (proudové a napěťové odezvy), 2D YZ řezy polí, 3D scatter ploty kinetiky částic i exporty časových řad do CSV.

---

## 2. Datový Workflow a HPC Paradigma

Simulace plazmatu ve 3D generují obrovské množství stavových dat. Aby bylo možné simulace spouštět dávkově na výpočetních clusterech bez grafického rozhraní, platí striktní workflow přes disk:

1. **Konfigurace (JSON)**: Parametry jsou externalizovány (`config.json` nebo `config_lab.json`).
2. **Geometrie a Vícevodičové Pozadí**:
   - Pokud je zadán soubor pozadí ze SPISu, je interpolován na mřížku.
   - Pokud soubor pozadí chybí (`spis_background_potential_file: ""`), pozadí $V_{bg}(\vec{r})$ a $\vec{E}_{bg}(\vec{r})$ je přesně syntetizováno lineární superpozicí Laplaceova operátoru trupu sondy a všech $N_{ant}$ antén:
     $$V_{bg}(\vec{r}) = V_{sc} \cdot V_{w, body}(\vec{r}) + \sum_{k=1}^{N_{ant}} V_{ant, k} \cdot V_{w, ant_k}(\vec{r})$$
3. **Výpočet a Decimace paměti**:
   - **Časová decimace** (`save_interval`): Ukládání stavu každých $N$ kroků.
   - **Prostorová decimace** (`plot_stride`): Ukládání reprezentativního vzorku částic pro 3D trajektorie.
4. **Serializace (`dust_impact.common.io`)**: Uložení komprimovaného binárního archivu `.npz` výhradně do složky `outputs/` (`outputs/out_vysledky.npz`).
5. **Lazy Loading a Vykreslení**: Vizualizační modul `plotting.py` čte data výhradně z `.npz` archivu bez nutnosti re-simulace.

---

## 3. Klíčová fyzikálně-numerická řešení

### 3.1. Exaktní geometrie antén a voxelizace vodičů ($O(1)$ kolize)
- **Princip**: Z váhového pole $V_w$ se extrahuje skutečný kovový objem anténního vodiče ($V_w \ge 0.85$ nebo obrysem `mesh.contour`).
- **Odstranění umělého nafukování**: Bylo zcela zrušeno dřívější umělé rozšiřování prostoru okolo drátu (cKDTree s poloměrem $15\text{ cm}$). Do výpočtu záchytu vstupuje výhradně skutečný průřez vodiče.
- **Ekvipotenciální těleso**: Uvnitř buněk kovu trupu sondy i antén je pevně nastaveno $V = V_0, V_w = 1.0$ a $\vec{E} = 0$.
- **Rychlost**: Detekce pohlcení částice probíhá v čase $O(1)$ přes indexování v matici `antenna_masks_3d[a_idx][i, j, k]`.

### 3.2. Rychlá 3D Trilineární Interpolace (Grid-to-Particle)
Vlastní vektorizovaná funkce `_interp_3d_fast` provádí trilineární interpolaci z 8 sousedních uzlů buňky s bezpečnostním ořezáním indexů (`np.clip`), což eliminuje knihovní režii univerzálních interpolátorů.

### 3.3. Vysoce výkonný Poissonův řešič (PyAMG SPD CG)
Pro řešení vlastní prostorové nábojové hustoty plazmatu $\rho$:
- **Formulace**: Soustava je zapsána jako Symetrická Pozitivně Definitivní matice ($-\nabla^2 V = \rho / \varepsilon_0$) s kladnou diagonálou.
- **Algebraic Multigrid (PyAMG)**: Využívá metodu sdružených gradientů (`accel='cg'`) s AMG předpodmiňovačem a teplým startem (předchozí řešení $x_0$). Konverguje za $< 0.05\text{ s}$ na mřížkách $150^3$ ($3.375\times 10^6$ uzlů).
- **Vektorizované sestavení RHS**: Vektor pravé strany $b = \text{np.nan\_to\_num}(\rho / \varepsilon_0)$ je generován blokově v NumPy bez Python `for` cyklů.

### 3.4. Duální režim injekce plazmatu (`plasma_injection_mode`)
1. **`"point_cloud"`**:
   - Vzniká v bodě dopadu na povrchu sondy $\vec{r}_{impact}$ s počáteční disperzí $\sigma = 0.01 \Delta x$.
   - Rychlosti jsou orientovány do poloprostoru podél normály povrchu $\vec{n}$ s termální Maxwellovskou složkou $v_{th} = \sqrt{k_B T / m_{ion}}$.
   - Makronáboj $q_{macro} = Q_{tot} / N_{macro}$ je určen škálovacím zákonem $Q_{tot} \propto m_{dust}^\alpha v_{dust}^\beta$.
2. **`"homogeneous"`**:
   - Homogenní náhodné rozmístění částic v celém výpočetním boxu $[-L, L]^3$ mimo kovové těleso sondy.
   - Rychlosti mají 3D izotropní Maxwellovské rozdělení.
   - Makronáboj $q_{macro} = e \cdot n_{sw} \cdot V_{dom} / N_{macro}$ odpovídá reálné hustotě okolního plazmatu slunečního větru.

### 3.5. Ramo-Shockleyho indukovaný proud a RC elektronika
- **Indukovaný proud**: $I_{ind, k} = -\sum q_{macro} (\vec{v} \cdot \vec{E}_{w, k})$, kde váhové elektrické pole $\vec{E}_{w, k} = -\nabla V_{w, k}$ má jednotky $[\text{m}^{-1}]$.
- **Dopadový proud**: $I_{col, k} = \frac{\Delta Q_k}{\Delta t}$ při pohlcení s efektivitou $\eta_{col}$.
- **RC Odezva**: Celkový proud $I_{tot} = I_{ind} + I_{col}$ je filtrován diferenciálním RC obvodem $\tau = R C$.

---

## 4. Spuštění a verifikace

```bash
# Spuštění simulace s konfiguračním souborem
python main.py --config config_lab.json --no-visualize

# Spuštění kompletní automatizované testovací sady (22 unit testů)
python -m unittest discover -s tests
```
