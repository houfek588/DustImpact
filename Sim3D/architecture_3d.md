# Architektura a logika 3D PIC Simulátoru pro AI Agenty

Tento dokument definuje systémovou architekturu, datový workflow a fyzikálně-numerická specifika 3D simulátoru hyperrychlých dopadů prachu.

Tento simulátor byl navržen pro analýzu signálů na vesmírných sondách (např. Solar Orbiter), kde dopad prachové částice rychlostí desítek km/s generuje oblak hustého plazmatu. Expanze tohoto mraku způsobuje komplexní elektromagnetické rušení a sběr náboje na detekčních anténách.

- Cíl tohoto dokumentu: Sloužit jako vyčerpávající referenční příručka (systémový kontext) pro umělou inteligenci a vývojáře, kteří budou kód v budoucnu rozšiřovat, optimalizovat, ladit, nebo přenášet na jinou výpočetní architekturu.

## 1. High-Level Architektura (Návrhový vzor)

Systém je striktně modulární a od počátku odděluje fyzikální výpočty (backend) od uživatelského rozhraní a vykreslování (frontend). Architektura fúzuje dva programovací přístupy:

- Data-Oriented Design (DOD): Použitý v horkých smyčkách (hot loops) uvnitř výpočetního jádra pro dosažení maximální propustnosti procesoru a optimálního využití L1/L2 cache (operace nad velkými souvislými bloky Numpy polí).
- Object-Oriented Programming (OOP): Použitý pro vnější zapouzdření stavů, správu životního cyklu simulace a udržení čistého jmenného prostoru (namespaces).

Základní moduly a jejich role:

- main_3d.py (Orchestrátor): Hlavní spouštěcí bod (entry point). Nemá žádnou vlastní fyzikální logiku. Řídí striktní lineární workflow: načtení konfigurace $\rightarrow$ inicializace výpočtu $\rightarrow$ běh simulace $\rightarrow$ uložení surových dat na disk $\rightarrow$ načtení z disku $\rightarrow$ spuštění vizualizační pipeline. Umožňuje snadné přepínání mezi režimem "jen počítej" a "jen vykresluj".
- input_data_3d.py (Data & Config Layer): Komplexní I/O rozhraní. Zajišťuje robustní načítání parametrů (automaticky generuje chybějící JSON konfigurace pomocí struktur dataclasses). Obsahuje logiku pro interakci se softwarem SPIS: importuje nestrukturované sítě .vtk pomocí knihovny PyVista, interpoluje je na pravoúhlý grid, vypočítává gradienty polí a provádí 3D voxelizaci. Dále obsahuje I/O rutiny pro efektivní binární kompresi do/z .npz.
- sim_core_3d.py (Compute Layer): Srdce celého systému. Samostatný objekt PIC (Particle-in-Cell) simulátoru. Zcela nezávislý na GUI, přijímá numpy pole a vrací numpy pole. Je odpovědný za časovou integraci (Leapfrog), řešení Poissonovy rovnice a interakci částic s tělesy (kolize a indukce proudu).
- plotting_3d.py (Presentation Layer): Prezentační vrstva. Přijímá vyčištěnou výpočetní historii a generuje statické grafy (matplotlib) i interaktivní 3D animace. Odpovídá i za případný export do formátů pro další zpracování (např. do .csv pro Excel nebo MATLAB).

## 2. Datový Workflow a HPC Paradigma

Simulace plazmatu ve 3D generují obrovské množství stavových dat (gigabajty dat každou vteřinu výpočtu). Aby bylo možné simulace spouštět dávkově na výpočetních clusterech (HPC - High Performance Computing) bez grafického rozhraní, platí striktní workflow přes pevný disk. Žádná data se nepředávají z jádra rovnou do grafů!

- Konfigurace (JSON): Parametry jsou externalizovány. Systém se nekonfiguruje změnou kódu, ale čtením z config_3d.json. To umožňuje snadné psaní wrapper skriptů pro parametrické studie (např. smyčka spouštějící simulaci pro různé polohy dopadu).

- Geometrie a Inicializace (VTK fallback): Importována jsou makroskopická pole (pozadí $V_{bg}$, váhová pole antén $V_w$) ze SW SPIS. Pokud selže import (soubory neexistují, mají špatný název, nebo obsahují neočekávané pojmenování datové vrstvy), systém automaticky nageneruje syntetická analytická pole (např. Gaussovské funkce). Tento robustní fallback garantuje, že algoritmus nikdy nespadne předčasně a umožní ladit kód i bez přítomnosti reálných objemných datových sad.

- Výpočet a Decimace paměti: sim_core_3d.py iteruje v časových krocích v řádech nanosekund (desítky tisíc kroků). Uložení poloh statisíců částic v každém kroku by zahltilo RAM. Proto se provádí agresivní decimace:

- Časová decimace (save_interval): Do historie se ukládá stav jen každý N-tý krok.

- Prostorová decimace (plot_stride): Ukládají se polohy jen reprezentativního vzorku částic (např. každé 10. částice).

- Serializace (NPZ Archiving): Kompletní výsledek (1D časové řady proudů a napětí + 4D historie částic a 3D matic polí) je komprimován a uložen jako binární archiv out_3d_vysledky.npz. Tento formát nativní pro Numpy dosahuje obrovské rychlosti I/O operací a excelentní komprese, čímž deklasuje formáty jako JSON nebo prosté CSV. HDF5 byl zvážen, ale zavržen kvůli nutnosti externích závislostí.

- Deserializace a Vykreslení: Vizualizační modul plotting_3d.py čte VŽDY výhradně paměťově načtený .npz soubor. Díky tomuto oddělení může vědec spustit 5hodinovou simulaci přes noc, uložit výsledky a druhý den bleskově otevírat, upravovat a exportovat grafy bez nutnosti přepočítávání celého fyzikálního modelu.

## 3. Klíčová fyzikálně-numerická řešení

Srdce PIC modelu využívá silně optimalizované numerické metody. Při budoucích úpravách kódu AI agenty je naprosto kritické pochopit a respektovat následující implementační principy. Narušení těchto principů by vedlo k exponenciálnímu zpomalení běhu programu (prokletí dimenzionality ve 3D).

### 3.1. Voxelizace pro O(1) detekci kolizí

- Problém: Sonda modelovaná ve SPISu se skládá ze statisíců trojúhelníků a složitých křivek. Pokud by algoritmus v každém nanosekundovém kroku testoval stovky tisíc letících makročástic na srážku s těmito polygony (metodami ray-castingu nebo průnikem bounding-boxů), simulace by se prakticky zastavila.

- Řešení: Využíváme metodu voxelizace (převod do rastru). V modulu input_data_3d.py se vyčte hraniční potenciál z VTK souboru a "obtiskne" se do 3D pravoúhlé boolovské numpy matice. Výsledkem je 3D pole jedniček a nul (např. mask = V_bg > limit).

- Implementace v horké smyčce: Spojitá plovoucí souřadnice částice $(x, y, z)$ se ořízne, zaokrouhlí na celé číslo a použije se přímo jako index. Pouhým dotazem do matice is_inside = mask[i, j, k] algoritmus okamžitě, v neměnném čase $O(1)$, odhalí, zda se částice právě nachází v kovu, a pokud ano, absorbuje ji.

### 3.2. Rychlá trilineární interpolace (Grid-to-Particle)

- Problém: Pole (elektrická síla) je diskretizováno v uzlech 3D sítě, avšak částice se pohybují spojitě "mezi" uzly v nekonečném prostoru $\mathbb{R}^3$. Použití univerzálních interpolátorů jako scipy.interpolate.RegularGridInterpolator je uvnitř cyklu neúnosně pomalé kvůli dodatečnému overheadu knihoven.

- Řešení: Jádro simulace (sim_core_3d.py) používá vlastní hardcoded vektorizovanou metodu _interp_3d_fast. Ta vypočítá 8 nejbližších sousedních uzlů, zjistí relativní vzdálenosti částice v rámci buňky (váhy $t_x, t_y, t_z$) a sečte je jediným průchodem přes Numpy pole. Rozšíření systému o jakékoliv nové fyzikální pole absolutně vyžaduje použití této, a žádné jiné, interpolační funkce.

### 3.3. Řešič Poissonovy rovnice a Dirichletovy okrajové podmínky

Elektrostatické chování plazmatu je řízeno Poissonovou rovnicí $\nabla^2 V = -\frac{\rho}{\varepsilon_0}$. Ve 3D mřížce (např. 40x40x40 uzlů) představuje rovnice obří soustavu 64 000 lineárních rovnic s 64 000 neznámými.

- Optimalizace (LU dekompozice): Matice soustavy $A$ (tzv. Laplacián ve 3D) závisí pouze na geometrii mřížky a tvarech sondy, které se během simulace nemění. Matice $A$ se proto sestaví a plně invertuje (faktorizuje) pouze jedinkrát, v konstruktoru __init__, pomocí scipy.sparse.linalg.factorized. Za běhu časového cyklu se používá již jen bleskurychlé maticové násobení $V = A^{-1}b$.

- Okrajové podmínky: Pevná napětí na plášti sondy a anténách (Dirichlet) jsou vynucena tak, že se na základě objevené Voxelové masky do hlavní Laplaciánovy matice $A$ vloží na diagonálu jednička a vektor pravé strany $b$ se v daném uzlu vynuluje.

### 3.4. Odezva elektroniky a Ramo-Shockleyho teorém

Makročástice na anténách sondy simulují fyzikální proud dvěma zcela odlišnými jevy, které algoritmus zpracovává souběžně a sčítá je:

- Indukovaný (Bezkontaktní) proud: Řídí se Ramo-Shockleyho teorémem. Vychází ze skalárního součinu rychlosti částice $\vec{v}$ a váhového elektrického pole $\vec{E}_w$. Pole $\vec{E}_w$ je derivací váhového potenciálu definovaného ze SPISu a představuje mapu citlivosti (kapacitní vazby) elektroniky na pohybující se náboje.

- Dopadový (Kolizní) proud: Vzniká čistou absorpcí. Když letící částice fyzicky protne voxelovou masku antény (viz bod 3.1), algoritmus hodí pomyslnou kostkou proti parametru collection_eff (efektivita sběru) a částici buď natrvalo pohlcuje do drátu (přičítá její náboj $q/\Delta t$ k celkovému proudu), nebo ji nechá se odrazit.
Oba proudy se sečtou a následně integrují do rovnice zpožděného paralelního RC obvodu (reprezentující reálný odpor zesilovačů a parazitní kapacitu měřáků sondy).

## 4. Konvence kódování pro budoucí AI iterace

Abychom předešli budoucí degradaci kódu (tzv. spaghetti code) a usnadnili LLM agentům pochopení logiky, musí každá nová úprava nebo refaktoring dodržet následující normy:

- Type Hinting (Typování): Vědecký kód v Pythonu je náchylný na chyby z důvodu těžkého využívání Numpy polí (je obtížné poznat, co je skalár, co vektor a co 3D matice). Každá nová i upravovaná metoda musí proto obsahovat striktní Python type hints (např. x: np.ndarray, active: np.ndarray, -> Tuple[np.ndarray, np.ndarray]). Pomáhá to automatickým linterům a AI analyzátorům odhalit "broadcast" chyby (nepasující dimenze).

- Numpy maskování (Zákaz for-cyklů pro částice): Algoritmus pracuje až s 1 000 000 částic. Pod hrozbou dramatického propadu výkonu je zakázáno iterovat přes částice klasickým for cyklem. Veškeré operace (filtrace mrtvých částic, posouvání, integrace) musí probíhat pomocí boolovských vektorových masek (např. zápisy typu new_active[new_active & (x <= 0)] = False).

- Správa paměti a logování: Při ukládání .npz archivu nebo při extrakci dat do histogramů je naprosto kritické zohlednit, že pole mají pevnou velikost definovanou na začátku a pohlcené/odletělé částice jsou nahrazeny hodnotou np.nan. Před aplikací analytických a kreslících funkcí (např. np.max) je nutné vždy aplikovat ~np.isnan() masku.

- Centralizace cest a konfigurací: Žádný soubor (kromě generátoru šablony) nesmí obsahovat takzvaná hardcoded (pevně zakódovaná) magická čísla nebo cesty ke složkám. Názvy všech VTK souborů, exportovaných obrázků, animačních formátů a cest k výsledným .npz musí být spravovány výhradně v konfiguračních třídách PlottingConfig3D a VTKFilesConfig.