# Architektura a logika 2D PIC Simulátoru pro AI Agenty

Tento dokument definuje ucelenou systémovou architekturu, datový workflow a fyzikálně-numerická specifika 2D simulátoru hyperrychlých dopadů prachových částic.

Fyzikální podstatou tohoto modelu je simulace událostí, při kterých prachová částice (mikrometeoroid) narazí kosmickou rychlostí (typicky 10 až 100 km/s) do trupu vesmírné sondy. Kinetická energie dopadu okamžitě vaporizuje a ionizuje část materiálu, čímž vzniká lokální oblak hustého, expandujícího plazmatu. Zatímco 1D model dokáže studovat pouze zjednodušené radiální šíření a 3D model vyžaduje složité a paměťově náročné zpracování externích CAD geometrií (např. ze SW SPIS), 2D model představuje dokonalý "zlatý střed" v oblasti kinetického modelování.

Je naprosto ideální pro rychlé, vysoce přesné parametrické studie při zachování extrémně rychlého výpočtu na běžných stolních počítačích. Ve 2D prostoru můžeme spolehlivě zkoumat komplexní jevy, jako jsou fázové posuny signálů, přeslechy mezi vícero anténami, asymetrie expanze při dopadech mimo středovou osu nebo vznik tzv. plazmatické stopy (wake effect) za překážkami.

Cíl tohoto dokumentu: Sloužit jako vyčerpávající referenční příručka (systémový kontext) pro umělou inteligenci a vývojáře, kteří budou kód v budoucnu rozšiřovat, optimalizovat, ladit nebo připravovat pro nasazení na výpočetní clustery.

## 1. High-Level Architektura (Návrhový vzor)

Systém je navržen striktně modulárně. Filosofií tohoto rozdělení je naprosté oddělení fyzikálních výpočtů (těžkého backendu) od uživatelského rozhraní, I/O operací a grafického vykreslování (pomalého frontendu). Architektura fúzuje dva odlišné programovací přístupy, aby maximalizovala jak výkon stroje, tak čitelnost kódu pro člověka:

- Data-Oriented Design (DOD): Tento přístup je striktně vynucován v tzv. horkých smyčkách (hot loops) uvnitř výpočetního jádra. Namísto vytváření tisíců objektů třídy Particle, což by vedlo k roztříštění paměti (cache misses), jsou data uložena jako velká, souvislá lineární Numpy pole (např. x_e, y_e, vx_e, active_e). Tento přístup umožňuje moderním procesorům využívat SIMD instrukce (Single Instruction, Multiple Data) a dosáhnout maximální propustnosti sběrnice a optimálního využití L1/L2 cache paměti.

- Object-Oriented Programming (OOP): Je využit pro vnější zapouzdření stavů makroskopických objektů, správu životního cyklu simulace a udržení čistého jmenného prostoru (namespaces). Třída DustImpactSimulation2D tak funguje jako černá skříňka, do které na začátku "vhodíme" parametry a na konci z ní "vypadnou" výsledky, aniž by došlo ke znečištění globálního prostoru.

### Základní moduly a jejich role:

- main_2d.py (Orchestrátor): Hlavní spouštěcí bod (pipeline manager). Záměrně neobsahuje vůbec žádnou fyzikální logiku ani algoritmy. Slouží jako dispečer, který řídí striktní lineární workflow: načtení konfigurace $\rightarrow$ inicializace výpočtu $\rightarrow$ běh časové smyčky $\rightarrow$ agregace surových dat $\rightarrow$ uložení na disk $\rightarrow$ načtení z disku $\rightarrow$ spuštění vizualizační pipeline. Umožňuje velmi snadno zakomentovat výpočetní část a používat skript pouze jako prohlížeč již spočítaných dat.

- input_data_2d.py (Data & Config Layer): Komplexní I/O a matematické rozhraní. Zajišťuje robustní načítání parametrů pomocí moderních struktur dataclasses. Pokud JSON konfigurace chybí, modul ji s výchozími hodnotami automaticky vygeneruje. Kromě datové správy modul sdružuje i analytické matematické předpisy pro 2D váhové funkce ($V_w, E_{wx}, E_{wy}$) odvozené z Gaussovy distribuce a obsahuje I/O rutiny pro binární kompresi a dekompresi .npz archivů.

- sim_core_2d.py (Compute Layer): Srdce celého systému a z hlediska CPU nejvytíženější modul. Jde o samostatný objekt PIC (Particle-in-Cell) simulátoru, který je programově izolovaný a nezávislý na GUI. Je odpovědný za iterativní časovou integraci (symplektický Leapfrog integrátor pro zachování energie), rozprostření náboje na mřížku (Particle-to-Grid), masivně paralelní řešení 2D Poissonovy rovnice a výpočet interakce částic se sadou libovolně rozmístěných antén (fyzické srážky a elektrostatická indukce proudu).

- plotting_2d.py (Presentation Layer): Prezentační a analytická vrstva. Přijímá vyčištěnou výpočetní historii a transformuje hrubá data do lidsky srozumitelné podoby. Generuje komplexní 2D vizualizace: termodynamické fázové prostory, 2D barevné heat-mapy z mřížkových polí, animované scatter ploty pro stopování makročástic a vyhlazené časové řady zaznamenaných proudů a napětí.

## 2. Datový Workflow a HPC Paradigma

Protože simulace plazmatu jsou přirozeně výpočetně velmi drahé, je systém od základu koncipován pro tzv. dávkové (batch) spouštění na výkonných stanicích. Platí proto striktní jednosměrný workflow přes pevný disk. Datové proudy nejsou nikdy předávány z výpočetního jádra přímo do vizualizačních funkcí bez mezikroku uložení na fyzické médium.

- Konfigurace (JSON): Veškeré běhové parametry (od velikosti boxu, přes integrační krok až po hmotnost prachu) jsou plně externalizovány do textového souboru config_2d.json. Tento přístup zaručuje reprodukovatelnost vědeckých experimentů. Zásadní inovací je to, že kód plně podporuje dynamický počet antén. V JSON konfiguraci jsou antény definovány jednoduše jako rozšiřitelný seznam (list) slovníků (s parametry polohy, odporu, kapacity a efektivity). Systém za běhu analyzuje strukturu JSONu a sám dynamicky alokuje velikost vícerozměrných polí pro libovolné množství senzorů.

- Geometrie a Inicializace (Analytická soběstačnost): Na rozdíl od své 3D sestřičky (která se spoléhá na těžkopádná data generovaná ve francouzském softwaru SPIS), 2D verze je záměrně navržena jako plně soběstačná (self-contained). Okrajové podmínky vodivého trupu sondy i prostorové distribuční profily váhových polí (simulující drátové antény) jsou generovány čistě matematicky, analyticky přímo v paměti za běhu. To činí tento kód mimořádně robustním, jelikož nevyžaduje žádnou zdlouhavou pre-processing přípravu datových sad a sítí.

- Výpočet a Agresivní Decimace Paměti: Kinetické jádro iteruje v extrémně jemných časových krocích (typicky v jednotkách nanosekund, simulujících celkový děj v řádu mikrosekund). Uložení kompletního stavu stovek tisíc částic v milionu kroků by znamenalo stovky gigabytů dat a pád systému na nedostatek RAM. Proto kód provádí chytrou decimaci na dvou úrovních:

- Časová decimace (save_interval): Úplný snímek historie se ukládá např. pouze každý 500. krok.

- Prostorová decimace (plot_stride): Ukládají se vizualizační polohy jen striktně reprezentativního vzorku částic (např. každá třicátá částice), zatímco makroskopická pole (matice potenciálu) se ukládají v plném mřížkovém rozlišení.

- Serializace (NPZ Archiving): Shromážděný výsledek simulace je komprimován a serializován pod kapotu formátu .npz (out_2d_vysledky.npz). Tento specifický formát vázaný na balíček Numpy byl zvolen kvůli své schopnosti dosáhnout bleskové I/O propustnosti (desítky milisekund) a minimální velikosti komprimovaného bloku na disku, čímž zásadně předčí lidsky čitelné, ale pomalé a objemné struktury jako CSV nebo JSON. Architektura udržuje plnou podporu dynamických antén – výstupy jako zaznamenané proudy se ukládají do tenzorových matic tvaru (počet_antén, počet_časových_kroků).

- Deserializace a Izolované Vykreslení: Modul plotting_2d.py funguje na principu "lazy loading". Čte a analyzuje VŽDY výhradně rozbalený .npz soubor. Pokud uživatel změní nastavení barev nebo vyžaduje vyrenderovat jiný typ grafu, proběhne to okamžitě nad načteným archivem, bez čekání na přepočet celého modelu. Modul navíc sám dynamicky adaptuje legendy, osy a počty vykreslených čar podle toho, kolik antén objeví ve vstupních datech.

## 3. Klíčová fyzikálně-numerická řešení

Srdce 2D PIC modelu stojí a padá na silně optimalizovaných numerických metodách z oblasti maticového počtu a lineární algebry. Porušení, nahrazení, či naivní refaktoring těchto návrhových vzorů (např. přepsáním vektorové operace do Python for-cyklu) by vedlo k exponenciálnímu zpomalení a razantnímu snížení výkonu.

### 3.1. Masivně paralelní Poissonův řešič (LU Faktorizace řídkých matic)

V metodě Particle-in-Cell je distribuce vlastního a vnějšího elektrostatického pole řízena Poissonovou parciální diferenciální rovnicí $\nabla^2 V = -\frac{\rho}{\varepsilon_0}$. V diskrétní 2D mřížce o rozměrech například $100 \times 100$ uzlů představuje diskretizace této rovnice obří soustavu $10\,000$ lineárních rovnic. Řešit toto v každém nanosekundovém kroku hrubou silou je pro CPU nereálné.

- Kritická optimalizace (Scipy Sparse & LU Dekompozice): Rovnice se modeluje pomocí centrálních konečných diferencí (tzv. 5-point stencil Laplacián). Kód sestaví matici systému $A$ v paměťově úsporném formátu velmi řídkých matic (scipy.sparse.lil_matrix). Zásadním krokem vpřed je uvědomění si faktu, že matice $A$ se v průběhu času nemění (mřížka je statická). Matice se tedy plně invertuje a rozloží na spodní a horní trojúhelníkovou matici pouze jedinkrát, během inicializace objektu v konstruktoru __init__, pomocí efektivní rutiny scipy.sparse.linalg.factorized. V každém dalším kroku simulace je pak drahý výpočet řešení rovnice zredukován na bleskurychlé a triviální maticové násobení napětí s nábojem $V = A^{-1}b$.

- Model inteligentně odděluje a řeší dvě matice zvlášť: jednu pro statické pole pozadí (kam aplikuje korekci pro plazmatické stínění dle Debyeovy délky k okrajům domény) a druhou čistě pro dynamické vlastní pole (self-field) expandujícího mraku, vypočtené z histogramu hustoty náboje. Tím zajišťuje fyzikální linearitu složení polí.

### 3.2. Rychlá Bilineární Interpolace a vážení (Grid-to-Particle)

- Problém: Numerická prostorová síť nabízí přesné hodnoty celkového elektrického pole ($\vec{E} = -\nabla V$) pouze ve svých fixních uzlech. Makroskopické částice se však pohybují volně, ve floatových desetinných souřadnicích $(x, y)$ v plném kontinuu.

- Řešení: Systém plně zavrhuje nativní Python metody typu scipy.interpolate.interp2d, jejichž overhead při milionech volání je fatální. Místo toho využívá vlastní, na míru napsanou, čistě Numpy-based vektorizovanou funkci _interp_field. Ta paralelně rozřadí polohu částice na nejbližší mřížkové indexy $i, j$ a vypočte zbytkové zlomky v buňce (váhy $tx, ty$). Finální efektivní hodnota působícího pole se získá váženým součtem 4 nejbližších mřížkových bodů rohů čtvercové buňky. To přináší oproti knihovním řešením zrychlení v řádech několika magnitud.

### 3.3. Analytická geometrie a dynamické maskování antén (Particle-to-Grid)

Systém ve 2D obejde nutnost vytváření objemných 3D boolovských voxelových masek (využívaných ve 3D verzi pro reprezentaci složitých konstrukcí ze SPISu). 2D analytické antény jsou definovány rychlou, exaktní matematickou podmínkou kružnice: (x - x_ant)^2 + (y - y_ant)^2 <= r_ant^2. To umožňuje plynule měnit poloměr detektorů pouhým přepsáním jednoho parametru v configu.

V inicializační fázi skript sestaví iterovatelný seznam fixních boolovských mřížkových masek self.ant_masks (pro každou anténu izolovaně) a také vygeneruje jednu sjednocenou globální masku self.combined_ant_mask.

Sjednocená maska je nepostradatelná pro správnou modifikaci Poissonovy rovnice – vkládá na tyto pozice Dirichletovy okrajové podmínky (vynucuje fixní potenciál kovu v mřížce). Separátní individuální masky se využívají uvnitř výpočetní smyčky _push_species ke zjištění, do které konkrétní antény daná makročástice narazila.

### 3.4. Ramo-Shockleyho teorém a zpožděná odezva RC elektroniky

Odezva signálu na detektorech je komplexní. Každá nakonfigurovaná anténa vyhodnocuje sběr a indukci naprosto izolovaně, čímž věrně simuluje samostatné kanály osciloskopu. Celkový proud vzniká superpozicí dvou odlišných jevů:

- Bezkontaktní Indukovaný Proud: Řídí se Ramo-Shockleyho teorémem. Pokud se náboj pohybuje v blízkosti vodiče, generuje v něm zpětný posuvný proud, aniž by došlo k fyzickému doteku. Tento jev je počítán jako efektivní 2D skalární součin vektoru rychlosti plazmatu a vektoru citlivosti antény: $I_{ind} = \sum q \cdot (v_x \cdot E_{wx} + v_y \cdot E_{wy})$. Tato metoda elegantně predikuje vznik známých bipolárních pulzů (když částice přilétá k anténě, indukuje jinou polaritu, než když od ní odlétá). Váhová pole se vyhodnocují extrémně efektivně za běhu prostým dosazením poloh do derivované Gaussovy obálky calc_Ew_2d.

- Přímý Dopadový Proud (Collection): Druhá, konvenčnější složka proudu vzniká de-ionizací fyzicky dopadlého náboje na povrch sondy. Pokud polohový vektor částice protne virtuální geometrii antény (přechod je detekován porovnáním současného stavu s paměťovým vektorem předchozího kroku was_outside), je vygenerován náhodný float z intervalu $(0, 1)$. Na základě porovnání s nastaveným prahem collection_efficiency v JSON configu je simulován tzv. Monte Carlo "hod kostkou", který rozhodne, zda je prachová plazmatická částice pohlcena (přičtena do proudu a smazána z PIC), nebo dojde k jejímu elastickému/neelastickému odrazu.

Získané komponentní proudy jsou v každém kroku $\Delta t$ sčítány do matice časových řad (tot_curr[a_idx, step]). Následně vstupují do algoritmu reprezentujícího diferenciální rovnici paralelního RC obvodu (simulujícího reálnou parazitní kapacitu a měřící odpor analogového převodníku). Integrující chování RC obvodu následně vyhlazuje a prodlužuje zjištěný napěťový puls do mikrosekundové odezvy, kterou sonda reálně naměří a odvysílá zpět na Zemi.

## 4. Konvence kódování pro budoucí AI iterace a úpravy lidmi

Architektura založená převážně na volně vázaných více-rozměrných typech v rámci Numpy se může při neopatrných zásazích rychle stát nečitelnou. Při jakémkoliv refaktoringu, přidávání nových modulů (např. magnetického pole) nebo modifikacích tohoto 2D kódu je z hlediska Clean Code bezpodmínečně nutné dodržet tyto dohodnuté konvence:

- Striktní Type Hinting (Statické Typování): Python je dynamicky typovaný jazyk, avšak v numerické vědě to často vede k nepříjemným "broadcast" chybám o desítky řádek dál. Každá nová metoda nebo signatura funkce musí obsahovat jasné Python type hints. Například: x: np.ndarray, popřípadě hist: Dict[str, Union[float, np.ndarray]] s přehlednou návratovou hodnotou -> Tuple[np.ndarray, np.ndarray]. Tento přístup dramaticky zvyšuje schopnost auto-complete funkcí a Linterů (např. MyPy) v IDE odhalit chybu před kompilací.

- Numpy vektorizace a Maskování (Zákaz izolovaných for-cyklů): Algoritmus zpracovává desetitisíce až statisíce částic ve statisících časových krocích. Iterace přes jednotlivé makroskopické částice pomocí klasického Python for cyklu je absolutně zakázána. Veškeré operace nad částicemi (aplikace urychlující síly, pohyb kinamatikou, smazání) se musí řešit výhradně blokově nad rozsáhlými sub-poli s využitím boolovských vyhodnocovacích masek. Příkladem čisté praxe je zápis: v_dot_Ew = vx[new_active] * Ewx + vy[new_active] * Ewy.

- Konzistentní zpracování neaktivních prvků (Hodnota np.nan): Kapacita (délka) stavových vektorů (x_e, y_e, ...) je neměnná a definovaná počtem emise na počátku. Když prachová částice dopadne na trup, absorbuje se do antény, nebo trvale opustí simulační okno (box), je jí programově odebrán flag z masky active. Do zaznamenávané historie self.history je nutné na tyto pozice v daném kroku exaktně vložit speciální plovoucí hodnotu np.nan (Not a Number), nikoliv hodnotu 0.0. Důvodem je, že hodnota 0.0 je plně legitimní fyzikální souřadnicí (např. střed sondy). Vykreslující moduly tak musí tyto hodnoty před aplikací matematických funkcí (min/max apod.) vždy odfiltrovat bezpečnou maskou ~np.isnan().

- Centralizace konfigurace a Nezávislost dat: Objekt orientovaný PIC kód simulátoru nesmí obsahovat tzv. "hardcoded" magická čísla nebo fyzikální parametry schované v útrobách metod. Variabilní dimenze mřížky, časové kroky či polohy a tloušťky drátů jsou plně zapouzdřeny a izolovaně poskytovány z aplikační třídy SimulationParams2D, která je de-serializována z formátu JSON. Plánujete-li přidat novou funkcionalitu (například volbu orientačního úhlu příletu prachu nebo hmotnosti jiného izotopu iontu), musíte logiku začít odvíjet vždy na začátku – nejprve rozšířit aplikační rozhraní (interface) v modulech a datových třídách input_data_2d.py, nikoliv nabouráním se dovnitř řešiče.