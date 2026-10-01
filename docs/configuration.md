# ⚙️ Konfigurace 3D PIC Simulátoru (`config.json`)

Tento dokument poskytuje detailní a ucelený popis všech parametrů konfiguračního JSON souboru pro 3D PIC simulátor **DustImpact (PostProcessSPIS)**.

Simulátor načítá konfiguraci buď z výchozího souboru [`config.json`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/config.json), nebo ze souboru předaného přepínačem `--config` (např. `python main.py --config inputs/config_analytical_sphere.json`).

---

## 📑 Přehled struktury konfiguračního souboru

Konfigurační JSON soubor je logicky rozdělen do 5 hlavních sekcí:

```
{
    "geometry": { ... },   // Definice geometrie (SPIS VTK sítě vs. Analytická tělesa)
    "toggles":  { ... },   // Booleovské přepínače fyzikálních modulů
    "physics":  { ... },   // Fyzikální vlastnosti plazmatu, náboje a dopadu (impact)
    "numeric":  { ... },   // Numerické parametry mřížky, časového kroku a integrace
    "plotting": { ... }    // Nastavení výstupů, ukládání (HDF5/NPZ), grafů a animací
}
```

---

## 1. Sekce `"geometry"`

Sekce `"geometry"` určuje, odkud a jakým způsobem se načítá nebo generuje prostorové uspořádání sondy, antén a elektrostatických polí.

### Klíčové parametry:

| Parametr | Typ | Výchozí hodnota | Popis |
|---|---|---|---|
| `source` | `string` | `"spis"` | Zdroj geometrie. Povolené hodnoty: `"spis"` (načtení z VTK polí ze SPISu) nebo `"analytical"` (čistě analytická tělesa bez nutnosti SPISu). |
| `spis` | `object` | `{ ... }` | Konfigurace pro SPIS variantu (použije se, když `source == "spis"`). |
| `analytical` | `object` | `{ ... }` | Konfigurace pro analytickou variantu (použije se, když `source == "analytical"`). |

---

### 1.1. Podsekce `"geometry"."spis"` (SPIS VTK geometrie)

Používá se pro načtení výsledků simulace ze softwaru SPIS ve formátu VTK:

```json
"geometry": {
    "source": "spis",
    "spis": {
        "background_potential_file": "inputs/spis_V_bg.vtk",
        "spacecraft": {
            "weighting_file": "inputs/spis_Vw_body.vtk",
            "surface_mesh_file": "",
            "voltage_V": 25.0
        },
        "antennas": [
            {
                "weighting_file": "inputs/spis_Vw_ant1.vtk",
                "voltage_V": 0.0,
                "capacitance_F": 2e-12,
                "resistance_Ohm": 100000.0
            }
        ],
        "weighting_threshold": 0.85
    }
}
```

| Parametr | Typ | Výchozí | Jednotka | Popis |
|---|---|---|---|---|
| `background_potential_file` | `string` | `"inputs/spis_V_bg.vtk"` | - | Cesta k VTK souboru s rovnovážným elektrostatickým potenciálem pozadí $V_{\text{bg}}$. Pokud je řetězec prázdný `""`, elektrostatické pozadí se automaticky vygeneruje superpozicí váhových polí trupu a antén. |
| `weighting_threshold` | `float` | `0.85` | - | Globální práh pro detekci/voxelizaci kovu tělesa sondy a antén z váhových polí $V_w$ ($V_w \ge \text{threshold}$). U tenkých antén kód obsahuje adaptivní fallback, pokud by mřížka byla příliš hrubá. |

#### Objekt `"spacecraft"` (Těleso sondy):
| Parametr | Typ | Výchozí | Jednotka | Popis |
|---|---|---|---|---|
| `weighting_file` | `string` | `"inputs/spis_Vw_body.vtk"` | - | Cesta k VTK souboru s Ramo-Shockleyho váhovým potenciálem tělesa sondy ($V_w = 1.0$ na povrchu sondy). |
| `surface_mesh_file` | `string` | `""` | - | Volitelná cesta k explicitní povrchové síti sondy (např. z Gmsh, STL nebo VTK). Pokud je prázdné, těleso se voxelizuje přímo z váhového pole. |
| `voltage_V` | `float` | `null` | $\text{V}$ | Rovnovážný plovoucí potenciál tělesa sondy $V_{\text{sc}}$. Pokud je `null`, načte se z výpočtu rovnovážného náboje nebo ze SPIS pozadí. |

#### Prvky pole `"antennas"`:
Každý prvek reprezentuje jednu měřicí anténu:
| Parametr | Typ | Výchozí | Jednotka | Popis |
|---|---|---|---|---|
| `weighting_file` | `string` | - | - | Cesta k VTK souboru s váhovým potenciálem dané antény ($V_w = 1.0$ na anténě, $0.0$ na ostatních vodičích). |
| `voltage_V` | `float` | `0.0` | $\text{V}$ | Počáteční předpětí (bias) antény vůči plazmatu / plovoucímu potenciálu. |
| `capacitance_F` | `float` | `2e-12` | $\text{F}$ | Vstupní kapacita $C_{\text{ant}}$ anténního předzesilovače a vedení (typicky $1\text{–}10\text{ pF}$). |
| `resistance_Ohm` | `float` | `100000.0` | $\Omega$ | Vstupní svodový odpor $R_{\text{ant}}$ anténního obvodu (typicky $100\text{ k}\Omega\text{ až }10\text{ M}\Omega$). |

---

### 1.2. Podsekce `"geometry"."analytical"` (Analytická geometrie)

Umožňuje provozovat plnohodnotné 3D simulace bez jakýchkoliv vstupů ze SPISu. Laplaceova pole se řeší automaticky vícedoménovým Poissonovým řešičem s PyAMG:

```json
"geometry": {
    "source": "analytical",
    "analytical": {
        "spacecraft": {
            "type": "sphere",
            "center": [0.0, 0.0, 0.0],
            "radius": 1.0,
            "voltage_V": 10.0
        },
        "antennas": [
            {
                "p_start": [0.0, 1.0, 0.0],
                "p_end": [0.0, 3.5, 0.0],
                "radius": 0.02,
                "voltage_V": 5.0,
                "capacitance_F": 2e-12,
                "resistance_Ohm": 100000.0
            }
        ]
    }
}
```

#### Objekt `"spacecraft"`:
| Parametr | Typ | Výchozí | Jednotka | Popis |
|---|---|---|---|---|
| `type` | `string` | `"sphere"` | - | Tvar tělesa sondy. Možnosti: `"sphere"` (koule), `"box"` / `"cube"` (kvádr/krychle), `"cylinder"` (válec), `"composite"` (složené těleso z více částí). |
| `center` | `list[3]` | `[0.0, 0.0, 0.0]` | $\text{m}$ | Souřadnice středu tělesa $[x_0, y_0, z_0]$. |
| `radius` | `float` | `1.0` | $\text{m}$ | Poloměr koule nebo válce (při `type: "sphere"` či `"cylinder"`). |
| `dimensions` | `list[3]` | `[1.0, 1.0, 1.0]` | $\text{m}$ | Délky stran kvádru $[a, b, c]$ podél os $X, Y, Z$ (při `type: "box"`). |
| `voltage_V` | `float` | `0.0` | $\text{V}$ | Elektrostatický potenciál tělesa sondy $V_{\text{sc}}$. |

#### Prvky pole `"antennas"`:
| Parametr | Typ | Výchozí | Jednotka | Popis |
|---|---|---|---|---|
| `p_start` | `list[3]` | `[0.0, 0.0, 0.0]` | $\text{m}$ | Počáteční bod osy válcové antény $[x_1, y_1, z_1]$ (typicky bod na povrchu sondy). |
| `p_end` | `list[3]` | `[1.0, 0.0, 0.0]` | $\text{m}$ | Koncový bod osy antény $[x_2, y_2, z_2]$ (hrot antény). |
| `radius` | `float` | `0.015` | $\text{m}$ | Fyzický poloměr válcového vodiče antény (např. $0.02 = 2\text{ cm}$). |
| `voltage_V` | `float` | `0.0` | $\text{V}$ | Elektrické předpětí (bias) dané antény. Pokud je parametr vynechán, nastaví se automaticky $0.0\text{ V}$. |
| `capacitance_F` | `float` | `2e-12` | $\text{F}$ | Kapacita RC obvodu antény. |
| `resistance_Ohm` | `float` | `100000.0` | $\Omega$ | Odpor RC obvodu antény. |

---

## 2. Sekce `"toggles"`

Booleovské přepínače, které umožňují izolovat jednotlivé fyzikální mechanismy a testovat jejich vliv na celkový signál:

```json
"toggles": {
    "enable_spis_background_field": true,
    "enable_plasma_self_field": true,
    "enable_antenna_particle_collection": true,
    "enable_rc_circuit_response": true,
    "enable_antenna_bias_voltage": true
}
```

| Přepínač | Typ | Výchozí | Popis |
|---|---|---|---|
| `enable_spis_background_field` | `bool` | `true` | Započtení statického pozadí elektrického pole $\vec{E}_{\text{bg}} = -\nabla V_{\text{bg}}$ do sil působících na částice (Newtonovy pohybové rovnice). |
| `enable_plasma_self_field` | `bool` | `true` | Zapnutí Poissonova řešiče pro výpočet vlastního prostorového náboje expandujícího plazmatu ($-\nabla^2 V_{\text{self}} = \rho / \varepsilon_0$). Při `false` částice expandují balisticky bez Coulombovského stínění. |
| `enable_antenna_particle_collection` | `bool` | `true` | Zapnutí absorpce elektronů a iontů povrchem antény a generování sběrového proudu $I_{\text{col}}(t)$. |
| `enable_rc_circuit_response` | `bool` | `true` | Zapnutí časové integrace RC obvodu antény ($\frac{dV}{dt} = \frac{I_{\text{tot}}}{C} - \frac{V}{RC}$). Při `false` se proud pouze zaznamenává, ale napětí se neintegruje. |
| `enable_antenna_bias_voltage` | `bool` | `true` | Inicializace počátečního předpětí antén z konfigurace v čase $t = 0$. |

---

## 3. Sekce `"physics"`

Obsahuje materiálové a plazmatické parametry slunečního větru, impaktního mraku a charakteristiku dopadu mikrometeoroidu.

```json
"physics": {
    "ion_mass_amu": 27.0,
    "impact_cloud_temperature_eV": 2.0,
    "solar_wind_electron_temp_eV": 15.0,
    "solar_wind_density_m3": 10000000.0,
    "total_impact_charge_C": 5e-11,
    "plasma_injection_mode": "point_cloud",
    "impact": {
        "location": [-2.0, 2.0, 0.0],
        "direction": [1.0, -1.0, 0.0],
        "time_delay_s": 1e-08
    }
}
```

| Parametr | Typ | Výchozí | Jednotka | Popis |
|---|---|---|---|---|
| `ion_mass_amu` | `float` | `27.0` | $\text{amu}$ | Hmotnost iontů vzniklých ionizací impaktního terče/prachu (např. $27.0$ pro hliník $\text{Al}$, $56.0$ pro železo $\text{Fe}$, $12.0$ pro uhlík $\text{C}$). |
| `impact_cloud_temperature_eV` | `float` | `2.0` | $\text{eV}$ | Počáteční kinetická teplota expandujícího impaktního plazmatu (typicky $1\text{–}5\text{ eV} \approx 10\,000\text{–}50\,000\text{ K}$). |
| `solar_wind_electron_temp_eV` | `float` | `15.0` | $\text{eV}$ | Teplota elektronů okolního slunečního větru (na $1\text{ AU}$ typicky $10\text{–}15\text{ eV}$). |
| `solar_wind_density_m3` | `float` | `1e7` | $\text{m}^{-3}$ | Číselná hustota okolního plazmatu (na $1\text{ AU}$ typicky $5\times 10^6\text{ až }10^7\text{ m}^{-3}$). Určuje Debyeovu délku $\lambda_D$. |
| `total_impact_charge_C` | `float` | `5e-11` | $\text{C}$ | Celkový kladný (i záporný) náboj $Q_{\text{tot}}$ uvolněný při dopadu prachového zrna (např. $50\text{ pC} = 5\times 10^{-11}\text{ C}$). |
| `plasma_injection_mode` | `string` | `"point_cloud"` | - | Režim injekce plazmatu:<br>• `"point_cloud"`: expanze plazmatického obláčku z bodu dopadu na povrchu sondy.<br>• `"homogeneous"`: rovnoměrné naplnění domény plazmatem slunečního větru. |

#### Podobjekt `"impact"`:
Sjednocuje geometrické a časové parametry nárazu:
| Parametr | Typ | Výchozí | Jednotka | Popis |
|---|---|---|---|---|
| `location` | `list[3]` | `[-2.0, 2.0, 0.0]` | $\text{m}$ | Počáteční prostorové souřadnice prachového zrna před nárazem $[x, y, z]$. Odtud je paprsek trasován k sondě. |
| `direction` | `list[3]` | `[1.0, -1.0, 0.0]` | - | Směrový vektor letu prachového zrna $[v_x, v_y, v_z]$. Algoritmus ray-marchingu nalezne přesný průsečík s povrchem sondy a vypočte lokální normálu terče $\vec{n}$. Pokud je vektor nulový `[0, 0, 0]`, bod dopadu je převzat přímo z `location`. |
| `time_delay_s` | `float` | `1e-6` | $\text{s}$ | Časové zpoždění expanze oblaku po zahájení simulace (umožňuje ustálení obvodů nebo zachycení před-impaktního šumu). |

---

## 4. Sekce `"numeric"`

Definuje výpočetní síť, časovou diskretizaci a počet makročástic pro Particle-in-Cell (PIC) metodu:

```json
"numeric": {
    "num_macroparticles": 20000,
    "time_step_s": 2e-09,
    "num_time_steps": 250,
    "domain_half_length_m": [10.0, 10.0, 10.0],
    "grid_nodes": [150, 150, 150]
}
```

| Parametr | Typ | Výchozí | Jednotka | Popis |
|---|---|---|---|---|
| `num_macroparticles` | `int` | `20000` | - | Celkový počet superčástic (polovina elektrony, polovina ionty) reprezentujících náboj $Q_{\text{tot}}$. |
| `time_step_s` | `float` | `2e-9` | $\text{s}$ | Časový integrační krok $\Delta t$. Musí splňovat CFL podmínku stability pro elektrony ($\Delta t \le \Delta x / v_{\text{th}}$). |
| `num_time_steps` | `int` | `10000` | - | Celkový počet časových kroků simulace $N_{\text{steps}}$. Celková simulovaná fyzikální doba je $T = N_{\text{steps}} \cdot \Delta t$. |
| `domain_half_length_m` | `list[3]` | `[5.0, 5.0, 5.0]` | $\text{m}$ | Půlrozměry simulační domény $[L_x, L_y, L_z]$. Výpočetní doména pokrývá interval $[-L_x, L_x] \times [-L_y, L_y] \times [-L_z, L_z]$. |
| `grid_nodes` | `list[3]` | `[35, 35, 35]` | - | Počet uzlů pravoúhlé mřížky $[N_x, N_y, N_z]$. Prostorový krok je $\Delta x = \frac{2 L_x}{N_x - 1}$. |

---

## 5. Sekce `"plotting"`

Řídí běhové vizualizace, formáty ukládání dat a exporty pro post-processing:

```json
"plotting": {
    "run_physical_simulation": true,
    "show_interactive_gui_windows": false,
    "save_plots_to_disk": true,
    "export_csv_time_series": false,
    "show_currents": true,
    "show_fields_slice": true,
    "show_particles_3d": true,
    "show_velocity_anim": true,
    "output_format": "h5",
    "output_h5_filepath": "outputs/out_vysledky.h5",
    "output_npz_filepath": "outputs/out_vysledky.npz",
    "output_csv_filepath": "outputs/out_vysledky_simulace.csv",
    "enable_checkpointing": true,
    "checkpoint_interval_steps": 500,
    "export_vtk": false,
    "vtk_output_dir": "outputs/paraview_vtk",
    "file_currents": "outputs/out_3d_proudy_napeti.png",
    "file_fields_anim": "outputs/out_3d_animace_pole_potencial.gif",
    "file_particles_anim": "outputs/out_3d_animace_pozice_castic.gif",
    "file_velocity_anim": "outputs/out_3d_animace_rychlosti.gif"
}
```

| Parametr | Typ | Výchozí | Popis |
|---|---|---|---|
| `run_physical_simulation` | `bool` | `true` | Při `true` se provede plný PIC výpočet. Při `false` (či CLI `--plot-only`) se výpočet přeskočí a načtou se existující data z disku. |
| `show_interactive_gui_windows` | `bool` | `false` | Zda po dokončení otevřít interaktivní grafická okna Matplotlib (doporučeno `false` při běhu na serveru bez GUI). |
| `save_plots_to_disk` | `bool` | `true` | Zda uložit vygenerované statické grafy a animace ve formátu PNG/GIF do výstupní složky. |
| `export_csv_time_series` | `bool` | `false` | Zda uložit časové řady proudů a napětí všech antén do přehledného CSV souboru. |
| `show_currents` | `bool` | `true` | Generování grafu časového průběhu indukovaných a sběrových proudů a napětí na anténách. |
| `show_fields_slice` | `bool` | `true` | Generování GIF animace prostorového řezu potenciálem plazmatu. |
| `show_particles_3d` | `bool` | `true` | Generování 3D prostorové animace kinetiky makročástic. |
| `show_velocity_anim` | `bool` | `true` | Generování fázové animace rychlostních profilů částic ($v_x, v_y, v_z$). |
| `output_format` | `string` | `"h5"` | Formát ukládání hlavních stavových polí a částic: `"h5"` (HDF5 komprimovaný dataset – vysoce doporučeno) nebo `"npz"` (NumPy zip archiv). |
| `output_h5_filepath` | `string` | `"outputs/out_3d_vysledky.h5"` | Cesta k výstupnímu HDF5 souboru. |
| `output_npz_filepath` | `string` | `"outputs/out_3d_vysledky.npz"` | Cesta k výstupnímu NPZ souboru. |
| `output_csv_filepath` | `string` | `"outputs/out_3d_vysledky_simulace.csv"` | Cesta k výstupnímu CSV souboru. |
| `enable_checkpointing` | `bool` | `true` | Ukládání atomických průběžných stavů simulace (umožňuje zotavení po výpadku či vyčerpání paměti). |
| `checkpoint_interval_steps` | `int` | `500` | Počet časových kroků mezi uloženími průběžného checkpointu. |
| `export_vtk` | `bool` | `false` | Export kompletních časových řad polí a částic do ParaView formátu (`.pvd`, `.vti`, `.vtp`). |
| `vtk_output_dir` | `string` | `"outputs/paraview_vtk"` | Složka pro uložení ParaView VTK dat. |
| `file_currents` | `string` | `"outputs/out_3d_proudy_napeti.png"` | Cesta pro uložení výsledného grafu odezvy antén. |
| `file_fields_anim` | `string` | `"outputs/out_3d_animace_pole_potencial.gif"` | Cesta pro animaci pole potenciálu. |
| `file_particles_anim` | `string` | `"outputs/out_3d_animace_pozice_castic.gif"` | Cesta pro 3D animaci částic. |
| `file_velocity_anim` | `string` | `"outputs/out_3d_animace_rychlosti.gif"` | Cesta pro fázovou animaci rychlostí. |

---

## 6. Kompletní ukázkové konfigurace

V repozitáři jsou připraveny validované konfigurační vzory:

1. **[`config.json`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/config.json)** a **[`inputs/config_template.json`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/inputs/config_template.json)**:
   - Plná geometrie ze SPISu se 3 anténami a tělesem sondy.
2. **[`inputs/config_analytical_sphere.json`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/inputs/config_analytical_sphere.json)**:
   - Analytická kulová sonda ($R = 1.0\text{ m}, V_{\text{sc}} = 10\text{ V}$) se 4 symetrickými dipólovými anténami.
3. **[`inputs/config_analytical_box.json`](file:///C:/Projects/Phd/SPIS/PostProcessSPIS/inputs/config_analytical_box.json)**:
   - Analytická kvádrová sonda ($1.6\times 1.6\times 1.6\text{ m}, V_{\text{sc}} = 15\text{ V}$) se 3 kolmými anténami.
