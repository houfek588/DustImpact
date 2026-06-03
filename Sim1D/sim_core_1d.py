#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
MODUL B: Výpočetní jádro (Objektově orientované) - SimCore
=============================================================================
Tento modul představuje samotné fyzikální a numerické "srdce" 1D simulace
dopadu prachové částice (mikrometeoroidu) na stěnu vesmírné sondy.

HLAVNÍ PRINCIPY A ARCHITEKTURA:
-------------------------------
Kód je založen na kinetickém přístupu Particle-in-Cell (PIC). Plazma není
modelováno jako spojitá tekutina (magnetohydrodynamika by zde selhala,
protože systém je plně bezkolizní), ale jako soubor tzv. makročástic.
Každá makročástice reprezentuje gigantický shluk skutečných elektronů nebo
iontů se stejným poměrem q/m (náboj ku hmotnosti), což nám umožňuje
řešit Lorentzovu sílu v rozumném výpočetním čase.

FYZIKÁLNÍ FENOMÉNY ŘEŠENÉ V TOMTO JÁDŘE:
----------------------------------------
1. Ambipolární expanze: Těžké ionty (např. odpařený hliník z trupu sondy) a
   extrémně lehké elektrony se po výbuchu snaží letět odlišnými rychlostmi.
   To vede k oddělení prostorového náboje, vzniku ohromných elektrostatických
   polí, brzdění elektronů a urychlování iontů.
2. Debyeovské stínění: Sonda i anténa nesou svůj vlastní statický náboj
   (vyjádřený potenciály Vf a V_ant_bias), který je ovšem stíněn okolním
   klidovým plazmatem slunečního větru.
3. Bezkontaktní měření (Ramo-Shockley): K měření signálu na anténě dochází
   ještě dlouho před tím, než k ní plazma fyzicky dorazí, a to indukcí.
4. Dynamika fázového prostoru: Algoritmus obsahuje několik numerických
   integrátorů (Euler, Leapfrog, RK4) pro zkoumání stability a zachování
   energie v čase.
"""

import numpy as np
from typing import Tuple, Dict, Any

# Importy základních fyzikálních konstant (e, m_e, eps_0, atd.)
from global_const import *
# Importy datových tříd a váhových funkcí (Ew je derivace váhového potenciálu Vw)
from input_data_1d import calc_Ew, SimulationParams, SimulationToggles


class DustImpactSimulation:
    """
    Hlavní třída, která udržuje stav a řídí evoluci plazmatického oblaku.

    Třída uchovává paměťové vektory pro polohy (x) a rychlosti (v) obou druhů
    makročástic (elektronů a iontů). Zabezpečuje střídání dvou kroků PIC cyklu:
    1. Grid -> Particle: Interpolace pole na částice a posun částic (Push).
    2. Particle -> Grid: Rozřazení náboje na mřížku a řešení Poissonovy rovnice.
    """

    def __init__(self, params: SimulationParams, toggles: SimulationToggles):
        """
        Konstruktor simulace: Inicializuje fázový prostor, okrajové podmínky
        a připraví paměťové bloky pro ukládání diagnostických dat.

        Parametry:
        ----------
        params : SimulationParams
            Instancovaná datová třída obsahující kalibraci, rozměry domény,
            hmotnosti iontů, teplotu plazmatu v eV a krokování času (dt).
        toggles : SimulationToggles
            Logické přepínače pro plynulé zapínání a vypínání jednotlivých
            fyzikálních bloků (vhodné pro studijní a analytické účely).
        """
        self.p = params
        self.toggles = toggles

        # ---------------------------------------------------------------------
        # INICIALIZACE KINETICKÉHO STAVU (PARTICLES)
        # ---------------------------------------------------------------------
        # Částice vznikají na stěně sondy (x=0). Jejich počáteční rychlosti
        # jsou taženy z Gaussova (Maxwell-Boltzmannova) normálního rozdělení.
        # Absolutní hodnota (np.abs) reprezentuje fakt, že do nitra sondy
        # (x < 0) částice letět nemohou – jedná se o tzv. half-Maxwellian emisi.
        self.v_e = np.abs(np.random.normal(0, self.p.v_th_e, self.p.N_particles))
        self.v_i = np.abs(np.random.normal(0, self.p.v_th_i, self.p.N_particles))

        # Všechny částice startují na přesné souřadnici x = 0 (místo impaktu)
        self.x_e = np.zeros(self.p.N_particles)
        self.x_i = np.zeros(self.p.N_particles)

        # Aktivní masky: Vynikající pro správu paměti a vektorizaci v Numpy.
        # Částice, které opustí doménu nebo narazí do drátů antény, dostanou False.
        # Počáteční stav je False, částice "ožijí" až po dosažení času t_delay.
        self.active_e = np.zeros(self.p.N_particles, dtype=bool)
        self.active_i = np.zeros(self.p.N_particles, dtype=bool)

        # Masky pro detekci křížení s rovinou antény v daném časovém kroku.
        self.was_outside_e = np.zeros(self.p.N_particles, dtype=bool)
        self.was_outside_i = np.zeros(self.p.N_particles, dtype=bool)

        self.cloud_injected = False

        # ---------------------------------------------------------------------
        # VÝSLEDKOVÁ POLE PRO OBVODOVÉ VELIČINY (CURRENTS & VOLTAGE)
        # ---------------------------------------------------------------------
        # Pole uchovávají elektrickou historii simulace po jednotlivých krocích (dt).
        self.ind_curr_e = np.zeros(self.p.steps)
        self.ind_curr_i = np.zeros(self.p.steps)
        self.col_curr_e = np.zeros(self.p.steps)
        self.col_curr_i = np.zeros(self.p.steps)
        self.tot_curr = np.zeros(self.p.steps)
        self.voltage_ant = np.zeros(self.p.steps)

        # ---------------------------------------------------------------------
        # PROSTOROVÁ MŘÍŽKA (EULERIAN GRID FIELDS)
        # ---------------------------------------------------------------------
        # Zde jsou uchovávána makroskopická pole spočtená z distribuce náboje.
        self.E_self_grid = np.zeros(self.p.N_grid)
        self.V_self_grid = np.zeros(self.p.N_grid)
        self.rho_grid = np.zeros(self.p.N_grid)

        # Předvýpočet statického pozadí (Background field)
        # Toto pole se v čase nemění. Vzniká nabíjením sondy a případným předpětím antény.
        self.E_bg_grid = np.zeros(self.p.N_grid)
        self.V_bg_grid = np.zeros(self.p.N_grid)

        if self.toggles.enable_background_field:
            if getattr(self.toggles, 'enable_antenna_bias', True):
                # Řešení lineární Poisson-Boltzmannovy rovnice (Debyeovo stínění) s pevnými okraji
                # V(0) = Vf a V(x_antenna) = V_ant_bias
                i_ant = np.argmin(np.abs(self.p.x_grid - self.p.x_antenna))
                x_a = self.p.x_antenna
                lam_D = self.p.debye_length

                # OBLAST 1: Mezi sondou a anténou (x <= x_antenna)
                # Použijeme přesné analytické řešení s hyperbolickými funkcemi (sinh)
                x_reg1 = self.p.x_grid[:i_ant + 1]
                self.V_bg_grid[:i_ant + 1] = (
                        self.p.Vf * np.sinh((x_a - x_reg1) / lam_D) / np.sinh(x_a / lam_D) +
                        self.p.V_ant_bias * np.sinh(x_reg1 / lam_D) / np.sinh(x_a / lam_D)
                )

                # OBLAST 2: Za anténou do volného prostoru (x > x_antenna)
                # Potenciál klesá exponenciálně z hodnoty V_ant_bias do 0
                x_reg2 = self.p.x_grid[i_ant + 1:]
                self.V_bg_grid[i_ant + 1:] = self.p.V_ant_bias * np.exp(-(x_reg2 - x_a) / lam_D)
            else:
                # Pokud anténa nemá bias, je v prostoru jen pole od sondy
                self.V_bg_grid = self.p.Vf * np.exp(-self.p.x_grid / self.p.debye_length)

            # Elektrické pole získáme derivací výsledného potenciálu
            # E = -dV/dx (np.gradient provádí centrální diferenci, což přirozeně
            # uhladí případnou nespojitost na samotné anténě).
            self.E_bg_grid = -np.gradient(self.V_bg_grid, self.p.dx)

        # ---------------------------------------------------------------------
        # DIAGNOSTICKÁ HISTORIE (PRO ANIMACE A FÁZOVÝ PROSTOR)
        # ---------------------------------------------------------------------
        self.history = {
            'E': [], 'V': [], 'rho': [], 't': [],
            'x_e': [], 'x_i': [], 'v_e': [], 'v_i': []
        }

    def _solve_poisson_equation(self):
        """
        Řeší 1D Poissonovu rovnici přes celou simulační mřížku pro získání
        vlastního (self-consistent) elektrického pole expandujícího plazmatu.

        Fyzikální a matematický detail:
        -------------------------------
        Rovnice má tvar d^2V/dx^2 = -rho/eps_0, respektive dE/dx = rho/eps_0.
        Integrace probíhá numericky přes metodu konečných diferencí na mřížce.
        """
        if not self.cloud_injected or not self.toggles.enable_self_field:
            self.E_self_grid.fill(0)
            self.V_self_grid.fill(0)
            self.rho_grid.fill(0)
            return

        # 1. KROK: PŘIŘAZENÍ NÁBOJE (Particle-to-Grid)
        # ---------------------------------------------------------------------
        # Zjišťujeme kolik aktivních elektronů a iontů se fyzicky nachází
        # ve kterém buňce (binu) mřížky pomocí numpy histogramu.
        counts_e, _ = np.histogram(self.x_e[self.active_e], bins=self.p.bin_edges)
        counts_i, _ = np.histogram(self.x_i[self.active_i], bins=self.p.bin_edges)

        # Prostorová hustota náboje rho [C/m^3].
        # A_sim je "1D expanzní průřez", který redukuje plošný náboj a
        # zabraňuje generování nereálně silných polí typických pro 1D modely.
        self.rho_grid = (counts_i - counts_e) * (self.p.q_macro / self.p.A_sim) / self.p.dx

        # Reset pracovních polí
        self.E_self_grid.fill(0)
        self.V_self_grid.fill(0)

        # 2. KROK: ŘEŠENÍ POISSONOVY ROVNICE S OKRAJOVÝMI PODMÍNKAMI
        # ---------------------------------------------------------------------
        if getattr(self.toggles, 'enable_antenna_bias', True):
            # Zjištění nejbližšího bodu mřížky odpovídajícího poloze antény
            i_ant = np.argmin(np.abs(self.p.x_grid - self.p.x_antenna))

            # ====================================================================
            # OBLAST 2: OD KONCE PROSTORU PO ANTÉNU (x > x_antenna)
            # ====================================================================
            # Fyzikální předpoklad: V nekonečnu (L_domain) je elektrické pole
            # i potenciál mraku přibližně roven nule. Postupujeme směrem k anténě.
            for i in range(self.p.N_grid - 2, i_ant - 1, -1):
                # Gaussův zákon: dE = (rho / eps_0) * dx
                self.E_self_grid[i] = self.E_self_grid[i + 1] - (self.rho_grid[i] * self.p.dx / eps_0)
                # Potenciál: dV = -E * dx
                self.V_self_grid[i] = self.V_self_grid[i + 1] + (self.E_self_grid[i] * self.p.dx)

            # Okrajová korekce pro Oblast 2 (Vynucení V(x_antenna) = 0)
            # Aby anténa stála jako elektrická pevnost, kompenzujeme případnou chybu
            # lineárním povrchovým korekčním polem E_corr_2.
            V_error_2 = self.V_self_grid[i_ant]
            L_region2 = self.p.L_domain - self.p.x_grid[i_ant]
            E_corr_2 = V_error_2 / L_region2 if L_region2 > 0 else 0

            self.E_self_grid[i_ant:] += E_corr_2
            self.V_self_grid[i_ant:] -= E_corr_2 * (self.p.L_domain - self.p.x_grid[i_ant:])

            # Uložíme pole těsně vpravo od antény pro finální průměrování
            E_ant_right = self.E_self_grid[i_ant]

            # ====================================================================
            # OBLAST 1: OD ANTÉNY PO POVRCH SONDY (x < x_antenna)
            # ====================================================================
            # Zde integrujeme z pozice antény (kde jsme zafixovali V=0) k trupu sondy.
            self.E_self_grid[i_ant] = 0
            self.V_self_grid[i_ant] = 0

            for i in range(i_ant - 1, -1, -1):
                self.E_self_grid[i] = self.E_self_grid[i + 1] - (self.rho_grid[i] * self.p.dx / eps_0)
                self.V_self_grid[i] = self.V_self_grid[i + 1] + (self.E_self_grid[i] * self.p.dx)

            # Okrajová korekce pro Oblast 1 (Vynucení V(0) = 0)
            # Trup sondy má enormní kapacitu. Vytvoří se na něm indukovaný náboj,
            # který jakoukoli odchylku potenciálu od 0 ihned srovná.
            V_error_1 = self.V_self_grid[0]
            L_region1 = self.p.x_grid[i_ant] - self.p.x_grid[0]
            E_corr_1 = V_error_1 / L_region1 if L_region1 > 0 else 0

            self.E_self_grid[:i_ant + 1] += E_corr_1
            self.V_self_grid[:i_ant + 1] -= E_corr_1 * (self.p.x_grid[i_ant] - self.p.x_grid[:i_ant + 1])

            E_ant_left = self.E_self_grid[i_ant]  # Uložíme pole těsně vlevo

            # Matematická singularita na anténě vede k nespojitosti (skoku) v elektrickém
            # poli (odpovídá povrchovému náboji sigma/eps_0). Abychom zamezili numerickému
            # přeskakování částic (particle jitter), pole přímo na uzlu zprůměrujeme.
            self.E_self_grid[i_ant] = (E_ant_left + E_ant_right) / 2.0

        else:
            # ====================================================================
            # REŽIM "MĚKKÉ ANTÉNY" (Bez fixního potenciálu, řeší se celá doména)
            # ====================================================================
            for i in range(self.p.N_grid - 2, -1, -1):
                self.E_self_grid[i] = self.E_self_grid[i + 1] - (self.rho_grid[i] * self.p.dx / eps_0)
                self.V_self_grid[i] = self.V_self_grid[i + 1] + (self.E_self_grid[i] * self.p.dx)

            # Okrajová korekce (Vynucení pouze V(0) = 0 na sondě)
            V_error = self.V_self_grid[0]
            E_corr = V_error / self.p.L_domain

            self.E_self_grid += E_corr
            self.V_self_grid -= E_corr * (self.p.L_domain - self.p.x_grid)

    def _get_accel(self, x_active: np.ndarray, mass: float, phys_charge: float, E_self_grid: np.ndarray) -> np.ndarray:
        """
        Vypočítá okamžité kinematické zrychlení pro vybrané částice.

        Všechna makroskopická pole (pole pozadí i pole plazmatu) jsou z mřížky
        lineárně interpolována do přesných, spojitých pozic (x) samotných částic.
        Díky tomu se zrychlení (a = q*E/m) počítá naprosto spojitě a je zaručena
        konzistence mezi gridem a částicemi.
        """
        E_tot = np.zeros(len(x_active))

        # Společná interpolace obou polí na pozici částice (Grid-to-Particle)
        if self.toggles.enable_background_field:
            E_tot += np.interp(x_active, self.p.x_grid, self.E_bg_grid)

        if self.toggles.enable_self_field:
            E_tot += np.interp(x_active, self.p.x_grid, E_self_grid)

        return (phys_charge / mass) * E_tot

    def _integrate_motion(self, x: np.ndarray, v: np.ndarray, active: np.ndarray, mass: float, phys_charge: float,
                          E_self_grid: np.ndarray, dt: float):
        """
        ZAPOUZDŘENÝ NUMERICKÝ INTEGRÁTOR
        Tato metoda představuje "Srdce" kinetiky celého PIC algoritmu. Zajišťuje
        posun poloh a rychlostí s ohledem na zvolenou přesnost.

        Integrátory k dispozici:
        ------------------------
        1. EULER (Forward Euler):
           Nejjednodušší přístup (1. řád). Kumuluje fázovou chybu a dlouhodobě
           porušuje zachování energie (tzv. "numerické zahřívání plazmatu").
        2. LEAPFROG (Verlet scheme):
           Symplektický integrátor (2. řád). Rychlost a poloha jsou fázově posunuty
           o poloviční krok (dt/2). Je naprosto stěžejní pro plazmovou fyziku,
           protože striktně zachovává makroskopickou energii, ačkoli lokální trajektorie
           mohou kmitat. Je levný na výpočet.
        3. RK4 (Runge-Kutta 4th Order):
           Extrémně přesný integrátor (4. řád). Snižuje chybu na O(dt^4), ale vyžaduje
           čtyřnásobný počet vyhodnocení pole pro každý uzel, což ho činí drahým
           na procesorový čas.
        """
        integrator = self.p.integrator.lower()

        # Provádíme řezy polem (slicing) přes aktivní masku. Vektorové operace
        # na menším poli (ignorujeme mrtvé částice) šetří výkon Numpy procesoru.
        x_act = x[active]
        v_act = v[active]

        if integrator in ['euler', 'leapfrog']:
            # Metoda Leapfrog zde strukturálně vypadá stejně jako Euler. Důvodem
            # je, že posunutí fází o -dt/2 pro Leapfrog proběhlo již exkluzivně
            # v metodě _inject_cloud() hned při startu, takže zde už smyčky "sedí".
            a = self._get_accel(x_act, mass, phys_charge, E_self_grid)
            v_act += a * dt
            x_act += v_act * dt

        elif integrator == 'rk4':
            # Krok 1: Predikce ze startovní pozice
            k1v = self._get_accel(x_act, mass, phys_charge, E_self_grid) * dt
            k1x = v_act * dt

            # Krok 2: Predikce z poloviny kroku za použití k1
            k2v = self._get_accel(x_act + k1x / 2.0, mass, phys_charge, E_self_grid) * dt
            k2x = (v_act + k1v / 2.0) * dt

            # Krok 3: Rafinovaná predikce z poloviny kroku za použití k2
            k3v = self._get_accel(x_act + k2x / 2.0, mass, phys_charge, E_self_grid) * dt
            k3x = (v_act + k2v / 2.0) * dt

            # Krok 4: Predikce z úplného konce kroku za použití k3
            k4v = self._get_accel(x_act + k3x, mass, phys_charge, E_self_grid) * dt
            k4x = (v_act + k3v) * dt

            # Vážený průměr všech 4 gradientů minimalizující trunkační chybu
            v_act += (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
            x_act += (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0

        else:
            raise ValueError(f"Neznámá integrační metoda: {integrator}")

        # Zpětný zápis vyřešených sub-polí do primárního bloku paměti
        x[active] = x_act
        v[active] = v_act

    def _push_species(self, x: np.ndarray, v: np.ndarray, active: np.ndarray, was_outside: np.ndarray, mass: float,
                      phys_charge: float, macro_charge: float, E_self_grid: np.ndarray) -> Tuple[float, float]:
        """
        Obsluhuje interakční smyčku a elektrické proudy pro jeden druh částic (E nebo I).

        Kromě volání samotné integrace má tato metoda na starosti:
        - Detekci propustnosti antény (účinnost záchytu).
        - Ořezávání uletěných částic (Boundary conditions).
        - Výpočet indukovaného a nasbíraného proudu (Využívá se q_macro).
        """
        # Optimalizace: Pokud druh vymřel nebo uletěl, nevolej náročné funkce
        if not np.any(active):
            return 0.0, 0.0

        # 1. KROK: Integrace polohy a rychlosti
        self._integrate_motion(x, v, active, mass, phys_charge, E_self_grid, self.p.dt)

        col_curr = 0.0
        ind_curr = 0.0

        # 2. KROK: Dopadová sběrná logika na anténě
        # Pomocí proměnné was_outside zkoumáme logický jev: "Proťala částice v
        # tomto okamžiku hraniční rovinu?". Tento přístup je odolný i vůči velmi
        # rychlým částicím, které proskočí anténou během jediného timestepu.
        is_outside = x >= self.p.x_antenna
        crossed_ant = active & (was_outside != is_outside)

        if self.toggles.enable_antenna_collection and np.any(crossed_ant):
            # Monte Carlo selekce: Z částic křižujících anténu jich zachytíme
            # určité procento určené faktorem propustnosti (collection_efficiency).
            absorbed = crossed_ant & (np.random.rand(self.p.N_particles) < self.p.collection_efficiency)

            # Celkový dodaný náboj pochází z makročástic. Makročástice mohou nést
            # desítky pC, i když jsou jich fyzicky v mřížce jen jednotky.
            dQ_coll = np.sum(absorbed) * macro_charge
            col_curr = dQ_coll / self.p.dt  # Změna náboje v čase dává měřitelný proud (A)

            # Nasbírané částice předaly náboj materiálu antény a ze simulace mizí
            active[absorbed] = False

        # Uložení logického stavu pro příští iteraci
        was_outside[active] = x[active] >= self.p.x_antenna

        # Boundary removal (Odstranění z prostoru)
        active[active & (x <= 0)] = False  # Dopad zpět do trupu vesmírné sondy
        active[active & (x >= self.p.L_domain)] = False  # Uletění ven do prostoru

        # 3. KROK: Indukovaný bezkontaktní proud (Ramo-Shockley theorem)
        if np.any(active):
            # Každá částice "hýbající" se v blízkosti nabitých elektrod způsobuje
            # relokaci povrchového náboje v kovu antény.
            Ew = calc_Ew(x[active], self.p.x_antenna, self.p.w_width)
            ind_curr = np.sum(macro_charge * v[active] * Ew)

        return col_curr, ind_curr

    def _update_circuit(self, step: int):
        """
        Zpracovává celkový makroskopický proud a propouští jej přes
        paralelní RC filtr reprezentující vstupní impedanci přijímače přístroje.

        Řeší se Obyčejná Diferenciální Rovnice prvního řádu:
        d(V_ant)/dt = I_tot / C_ant - V_ant / (R_ant * C_ant)
        """
        # I_tot je prostým součtem všech přispívajících plazmatických komponent.
        I_tot = (self.ind_curr_e[step] + self.ind_curr_i[step] +
                 self.col_curr_e[step] + self.col_curr_i[step])
        self.tot_curr[step] = I_tot

        if step > 0:
            if self.toggles.enable_rc_circuit:
                dV_dt = (I_tot / self.p.C_ant) - (self.voltage_ant[step - 1] / (self.p.R_ant * self.p.C_ant))
            else:
                # Při vypnutém RC obvodu funguje systém jako dokonalý kapacitní integrátor
                dV_dt = I_tot / self.p.C_ant
            self.voltage_ant[step] = self.voltage_ant[step - 1] + dV_dt * self.p.dt

    def _save_history(self, current_time: float):
        """
        Agreguje časové "snímky" prostorových polí a distribucí makročástic.
        Kvůli prevenci vyčerpání operační paměti RAM je aplikován plot_stride,
        který ukládá např. pouze každou 20. makročástici pro tvorbu animací.
        """
        self.history['E'].append(self.E_bg_grid + self.E_self_grid)
        self.history['V'].append(self.V_bg_grid + self.V_self_grid)
        self.history['rho'].append(self.rho_grid.copy())
        self.history['t'].append(current_time)

        # np.where vkládá NaN pro vypnuté částice, aby se ve Scatter() grafech nevykreslovaly.
        hx_e = np.where(self.active_e, self.x_e, np.nan)[::self.p.plot_stride]
        hx_i = np.where(self.active_i, self.x_i, np.nan)[::self.p.plot_stride]
        self.history['x_e'].append(hx_e)
        self.history['x_i'].append(hx_i)

        hv_e = np.where(self.active_e, self.v_e, np.nan)[::self.p.plot_stride]
        hv_i = np.where(self.active_i, self.v_i, np.nan)[::self.p.plot_stride]
        self.history['v_e'].append(hv_e)
        self.history['v_i'].append(hv_i)

    def _inject_cloud(self):
        """
        Trigger pro započetí dynamické simulace.
        Kromě aktivace částicových masků zajišťuje specifickou fázovou
        iniciaci (zpoždění) vyžadovanou symplektickým Leapfrog integrátorem.
        """
        self.active_e[:] = True
        self.active_i[:] = True
        self.was_outside_e[:] = False
        self.was_outside_i[:] = False
        self.cloud_injected = True

        # LEAPFROG specifikum: Posunout rychlosti o polovinu časového kroku vzad.
        if self.p.integrator.lower() == 'leapfrog':
            self._solve_poisson_equation()
            a_e_init = self._get_accel(self.x_e[self.active_e], m_e, -e, self.E_self_grid)
            a_i_init = self._get_accel(self.x_i[self.active_i], self.p.m_i, e, self.E_self_grid)
            self.v_e[self.active_e] -= 0.5 * a_e_init * self.p.dt
            self.v_i[self.active_i] -= 0.5 * a_i_init * self.p.dt

    def run(self) -> Dict[str, Any]:
        """
        Nejvyšší řídící smyčka jádra (Main Loop).
        Zajišťuje krok-za-krokem (step-by-step) provádění fyzikální integrace
        prostoru a času. Koncipováno pro budoucí možnost integrace s ProgressBar.
        """
        print(f"Running simulation... Integrator: {self.p.integrator.upper()}")
        print(f"Active toggles: {self.toggles}")

        for step in range(self.p.steps):
            current_time = step * self.p.dt

            # Injection condition (Nastal čas simulovat impakt)
            if not self.cloud_injected and current_time >= self.p.t_delay:
                self._inject_cloud()

            # Physics loop (Vyhodnocení kolektivních elektrostatických interakcí)
            self._solve_poisson_equation()

            if self.cloud_injected:
                # Kinetika elektronů
                c_e, i_e = self._push_species(
                    self.x_e, self.v_e, self.active_e, self.was_outside_e,
                    m_e, -e, -self.p.q_macro, self.E_self_grid)
                self.col_curr_e[step] = c_e
                self.ind_curr_e[step] = i_e

                # Kinetika iontů
                c_i, i_i = self._push_species(
                    self.x_i, self.v_i, self.active_i, self.was_outside_i,
                    self.p.m_i, e, self.p.q_macro, self.E_self_grid)
                self.col_curr_i[step] = c_i
                self.ind_curr_i[step] = i_i

            # Elektrické obvody a ukládání telemetrie
            self._update_circuit(step)

            if step % self.p.save_interval == 0 or step == self.p.steps - 1:
                self._save_history(current_time)

        return self._post_process()

    def _post_process(self) -> Dict[str, Any]:
        """
        Hladítko a pakovač výsledků. V praxi nasbírané náboje dopadají v
        ostrých diskrétních pulsech. Tyto šumy jsou vyhlazeny plovoucím
        průměrem pro hladší prezentaci dat simulujících biliony skutečných částic.
        """
        smooth_window = 100
        kernel = np.ones(smooth_window) / smooth_window

        smooth_induced = np.convolve(self.ind_curr_e + self.ind_curr_i, kernel, mode='same')
        smooth_collected = np.convolve(self.col_curr_e + self.col_curr_i, kernel, mode='same')
        smooth_total = np.convolve(self.tot_curr, kernel, mode='same')

        return {
            'smooth_induced': smooth_induced,
            'smooth_collected': smooth_collected,
            'smooth_total': smooth_total,
            'voltage_ant': self.voltage_ant,
            'history': self.history
        }