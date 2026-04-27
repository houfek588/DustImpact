#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
MODUL C: Vykreslování, vizualizace a vědecká analýza dat
=============================================================================
Tento modul slouží jako komplexní prezentační, analytická a edukační vrstva pro
1D Particle-in-Cell (PIC) simulaci hyperrychlého dopadu prachové částice.

ARCHITEKTURA A CÍL MODULU:
Cílem tohoto modulu je striktně oddělit grafické výstupy a post-processing od
samotného výpočetního jádra (Modul B). Modul přijímá hrubá numerická data
(časoprostorové pozice, rychlosti částic, makroskopická pole a spočítané elektrické
proudy) a transformuje je do srozumitelných vědeckých grafů a dynamických animací.

FYZIKÁLNÍ DĚJE A JEJICH VIZUALIZACE:
Modul je rozdělen do několika specializovaných funkcí, z nichž každá analyzuje
a vizualizuje specifický fyzikální aspekt plazmatické expanze:

1. Makroskopické signály (Proudy a RC filtr):
   Ukazuje, jak se teoretické proudy (indukované a přímo nasbírané) transformují
   přes elektroniku přístroje do měřitelného napětí.

2. Makroskopická pole (Poissonova rovnice):
   Vizualizuje separaci náboje (lehké elektrony unikají rychleji než těžké ionty)
   a vznik silného ambipolárního elektrického pole, které drží oblak pohromadě.

3. Váhové pole (Ramo-Shockleyho věta):
   Vykresluje citlivost antény v prostoru, nutnou pro bezkontaktní indukci signálu.

4. Kinetika oblaku (Pozice makročástic):
   Přímý pohled na šíření rázové/expanzní vlny plazmatu a demonstrace
   absorpce částic při průletu propustnou mřížkou antény.

5. Termodynamika plazmatu (Rozložení rychlostí):
   Animované histogramy demonstrující přeměnu kinetické energie. Zobrazuje
   ambipolární brzdění elektronů a urychlování iontů z počátečního
   Maxwell-Boltzmannova rozdělení.

6. Fázový prostor (Phase Space, x vs. v):
   Komplexní pohled na dynamiku plazmatu vycházející z Vlasovovy rovnice,
   umožňující pozorovat termalizaci, tvoření shluků a děr ve fázovém prostoru.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, Optional, Tuple

# Importy z předchozích fyzikálních modulů (upravte podle skutečné struktury projektu)
from input_data_1d import calc_Ew, calc_Vw, SimulationParams


def _plot_currents_and_voltage(results: Dict[str, Any], params: SimulationParams,
                               plot_toggles: Dict[str, bool]) -> None:
    """
    Vygeneruje statický graf celkových proudů a výsledného napětí na anténě.

    Fyzikální kontext:
    ------------------
    Celkový proud generovaný plazmatem se skládá z "indukovaného" (pohyb náboje
    v okolí antény, který posouvá nosiče náboje v kovu antény na dálku) a
    "nasbíraného" proudu (přímá absorpce elektronů a iontů materiálem antény).
    Tento proud následně protéká přes reálnou elektroniku (vstupní impedanci
    přijímače a kapacitu samotné antény), která tvoří dolní propust (RC filtr).

    Technická implementace:
    -----------------------
    Vykreslují se dva podgrafy (proudy nahoře, napětí dole). Používají se vyhlazené
    řady (smooth_induced, atd.) z post-processingu jádra, aby se omezil statistický
    šum typický pro PIC simulace s konečným počtem makročástic.
    """
    if not plot_toggles.get('show_currents', True):
        return

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))
    fig1.canvas.manager.set_window_title('Makroskopické signály: Proudy a Napětí')

    # Podgraf A: Kinetické a dopadové proudy
    # Všechny proudy převádíme z Ampérů na nanoAmpéry (1e9) pro lepší čitelnost.
    ax1a.plot(params.time_array * 1e6, results['smooth_induced'] * 1e9, color='blue', lw=1.5,
              label='Indukovaný proud (Ramo-Shockley)')
    ax1a.plot(params.time_array * 1e6, results['smooth_collected'] * 1e9, color='red', lw=1.5,
              label='Nasbíraný proud (Fyzické dopady)')
    ax1a.plot(params.time_array * 1e6, results['smooth_total'] * 1e9, color='black', lw=2, linestyle=':',
              label='Celkový součtový proud')

    # Vyznačení kritického momentu - času dopadu a následného výbuchu prachu
    ax1a.axvline(x=params.t_delay * 1e6, color='grey', linestyle='--', alpha=0.7, label='Injekce prachu')

    ax1a.set_title(f"Plazmatické proudy generované částicemi u antény (Vzdálenost = {params.x_antenna} m)")
    ax1a.set_ylabel("Elektrický proud [nA]")
    ax1a.legend(loc='upper right')
    ax1a.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1a.minorticks_on()

    # Podgraf B: Odezva RC obvodu (řešení diferenciální rovnice provedené v jádře)
    # Časová konstanta tau určuje rychlost vybíjení kondenzátoru.
    tau_us = params.R_ant * params.C_ant * 1e6
    # Napětí převádíme z Voltů na milivolty (1e3).
    ax1b.plot(params.time_array * 1e6, results['voltage_ant'] * 1e3, color='green', lw=2.5,
              label=f'Měřené napětí na přijímači ($\\tau$ = {tau_us:.1f} µs)')
    ax1b.axvline(x=params.t_delay * 1e6, color='grey', linestyle='--', alpha=0.7)

    ax1b.set_title(
        f"Zpracování signálu RC filtrem (Vstupní odpor: {params.R_ant / 1000:.0f} kΩ, Kapacita antény: {params.C_ant * 1e12:.0f} pF)")
    ax1b.set_xlabel("Čas od začátku simulace [µs]")
    ax1b.set_ylabel("Indukované napětí [mV]")
    ax1b.legend(loc='upper right')
    ax1b.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1b.minorticks_on()

    fig1.tight_layout()

    if plot_toggles.get('save_plots', False):
        fig1.savefig('out_graf_proudy_napeti.png', dpi=300, bbox_inches='tight')
        print("  [OK] Uloženo: out_graf_proudy_napeti.png")


def _animate_macroscopic_fields(hist: Dict[str, Any], params: SimulationParams, plot_toggles: Dict[str, bool],
                                animations: list) -> None:
    """
    Vygeneruje dynamickou animaci elektrického pole, hustoty náboje a potenciálu.

    Fyzikální kontext:
    ------------------
    Tato funkce přímo vizualizuje řešení Poissonovy rovnice. Ukazuje klíčový
    fenomén plazmatu - kvazineutralitu a ambipolární difúzi. Rychlé elektrony
    se snaží uniknout, čímž vzniká lokální přebytek záporného náboje na čele vlny
    a přebytek kladného náboje vzadu u iontů (graf hustoty náboje). Tato separace
    generuje silné elektrické pole (graf E), které brání dalšímu rozdělování.
    Graf potenciálu navíc ukazuje vliv Dirichletovy okrajové podmínky (povrch sondy).
    """
    if not plot_toggles.get('show_fields_anim', True):
        return

    fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig3.canvas.manager.set_window_title('Makroskopická pole plazmatu')

    # Příprava osy pro Elektrické pole
    line_E, = ax1.plot([], [], lw=2, color='purple')
    ax1.set_xlim(0, params.L_domain)
    # Zjištění absolutních extrémů pro zafixování osy Y, aby graf při animaci nedýchal
    E_max, E_min = np.max(hist['E']), np.min(hist['E'])
    E_range = E_max - E_min if E_max != E_min else 1.0
    ax1.set_ylim(E_min - 0.1 * E_range, E_max + 0.1 * E_range)
    ax1.set_title("Vývoj celkového makroskopického elektrického pole")
    ax1.set_ylabel("Elektrické pole E [V/m]")
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Příprava osy pro Hustotu náboje
    line_rho, = ax2.plot([], [], lw=2, color='green')
    ax2.set_xlim(0, params.L_domain)
    rho_max = np.max(np.abs(hist['rho']))
    if rho_max == 0: rho_max = 1e-12
    ax2.set_ylim(-rho_max * 1.1, rho_max * 1.1)
    ax2.set_title("Prostorová hustota náboje (Oddělení elektronů a iontů)")
    ax2.set_ylabel("Hustota náboje $\\rho$ [C/m$^3$]")
    ax2.grid(True, linestyle=':', alpha=0.7)

    # Příprava osy pro Elektrický potenciál
    line_V, = ax3.plot([], [], lw=2, color='blue')
    ax3.set_xlim(0, params.L_domain)
    # Vyznačení antény červenými čarami ve všech podgrafech
    ax1.axvline(x=params.x_antenna, color='red', linestyle='-', alpha=0.6, lw=1.5)
    ax2.axvline(x=params.x_antenna, color='red', linestyle='-', alpha=0.6, lw=1.5, label='Poloha detekční antény')
    ax2.legend(loc='upper right')
    ax3.axvline(x=params.x_antenna, color='red', linestyle='-', alpha=0.6, lw=1.5)

    V_max, V_min = np.max(hist['V']), np.min(hist['V'])
    V_range = V_max - V_min if V_max != V_min else 1.0
    ax3.set_ylim(V_min - 0.1 * V_range, V_max + 0.1 * V_range)
    ax3.set_title("Elektrický potenciál plazmatu")
    ax3.set_xlabel("Vzdálenost od těla sondy x [m]")
    ax3.set_ylabel("Potenciál V [V]")
    ax3.grid(True, linestyle=':', alpha=0.7)

    # Textové pole pro aktuální čas běžící uvnitř animace
    time_text_fig3 = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, fontsize=11,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))

    def init_fig3():
        """ Inicializační funkce pro blit=True animaci (vrací prázdné objekty) """
        line_E.set_data([], [])
        line_rho.set_data([], [])
        line_V.set_data([], [])
        time_text_fig3.set_text('')
        return line_E, line_rho, line_V, time_text_fig3

    def animate_fig3(i):
        """ Aktualizuje data linek pro snímek 'i' """
        line_E.set_data(params.x_grid, hist['E'][i])
        line_rho.set_data(params.x_grid, hist['rho'][i])
        line_V.set_data(params.x_grid, hist['V'][i])
        time_text_fig3.set_text(f"Simulační čas: {hist['t'][i] * 1e6:.2f} µs")
        return line_E, line_rho, line_V, time_text_fig3

    # Vytvoření objektu animace. Používáme blit=True pro výrazné zrychlení překreslování.
    anim_fig3 = animation.FuncAnimation(fig3, animate_fig3, init_func=init_fig3, frames=len(hist['t']),
                                        interval=100, blit=True)
    animations.append(anim_fig3)
    fig3.tight_layout()

    if plot_toggles.get('save_plots', False):
        print("  -> Ukládám animaci polí...")
        anim_fig3.save('out_animace_pole_potencial.gif', writer='pillow', fps=15)
        print("  [OK] Uloženo: out_animace_pole_potencial.gif")


def _plot_weighting_field(params: SimulationParams, plot_toggles: Dict[str, bool]) -> None:
    """
    Vykreslí statický tvar váhového potenciálu a váhového pole.

    Fyzikální kontext:
    ------------------
    Ramo-Shockleyho teorém pracuje s fiktivním "váhovým polem". To je geometrická
    vlastnost dané sestavy elektrod (sonda a anténa), která popisuje, jak moc
    je náboj v určitém bodě prostoru schopen ovlivnit napětí na anténě.
    Zde je pole modelováno aproximací pomocí úzkého Gaussova zvonu, což znamená,
    že anténa je nejvíce citlivá na letící částice pouze ve své bezprostřední blízkosti.
    """
    if not plot_toggles.get('show_weighting_field', True):
        return

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    fig4.canvas.manager.set_window_title('Ramo-Shockley: Citlivost antény')

    ax4.plot(params.x_grid, calc_Vw(params.x_grid, params.x_antenna, params.w_width), lw=2, color='orange',
             linestyle='--',
             label='Váhový potenciál $V_w(x)$')
    ax4.plot(params.x_grid, calc_Ew(params.x_grid, params.x_antenna, params.w_width), lw=2, color='magenta',
             label='Váhové pole $E_w(x)$ (Derivace)')

    ax4.axvline(x=params.x_antenna, color='red', linestyle='-', alpha=0.6, lw=2,
                label='Fyzické umístění drátové antény')
    ax4.axhline(y=0, color='black', lw=1, alpha=0.5)

    ax4.set_title("Váhové funkce pro výpočet indukovaného proudu (Bezkontaktní měření)")
    ax4.set_xlabel("Vzdálenost x [m]")
    ax4.set_ylabel("Relativní amplituda citlivosti")
    ax4.legend(loc='upper left')
    ax4.grid(True, linestyle=':', alpha=0.7)
    fig4.tight_layout()

    if plot_toggles.get('save_plots', False):
        fig4.savefig('out_graf_vahove_pole.png', dpi=300, bbox_inches='tight')
        print("  [OK] Uloženo: out_graf_vahove_pole.png")


def _animate_particles(hist: Dict[str, Any], params: SimulationParams, plot_toggles: Dict[str, bool],
                       animations: list) -> None:
    """
    Animace 1D kinematiky obrovského množství makročástic.

    Fyzikální kontext:
    ------------------
    Zobrazuje polohu oblaku v reálném čase. Poskytuje vynikající intuici pro
    dopřednou rychlost vlny a znázorňuje stochastickou (Monte-Carlo) absorpci
    náboje na propustné anténě. Elektrony jsou vizuálně rozptýlené do "mraku",
    ačkoliv se fyzikálně pohybují po jedné 1D přímce.
    """
    if not plot_toggles.get('show_particles_anim', True):
        return

    fig5, ax_pos = plt.subplots(figsize=(10, 4))
    fig5.canvas.manager.set_window_title('Kinetika: Pozice makročástic')

    scat_e = ax_pos.scatter([], [], s=3, color='red', alpha=0.5, label='Makročástice elektronů', edgecolors='none')
    scat_i = ax_pos.scatter([], [], s=3, color='blue', alpha=0.5, label='Makročástice iontů', edgecolors='none')

    ax_pos.axvline(x=params.x_antenna, color='black', linestyle='--', lw=2,
                   label=f'Mřížka antény (Účinnost záchytu: {params.collection_efficiency * 100:.0f}%)')

    ax_pos.set_xlim(0, params.L_domain)
    ax_pos.set_ylim(0, 1)
    ax_pos.set_yticks([])  # Skrytí Y osy, protože nemá fyzikální význam
    ax_pos.set_title("1D Kinetika: Průlet plazmatického oblaku simulační doménou")
    ax_pos.set_xlabel("Vzdálenost od místa dopadu x [m]")
    ax_pos.legend(loc='upper right')

    time_text_pos = ax_pos.text(0.02, 0.85, '', transform=ax_pos.transAxes, fontsize=11,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))

    # Umělý vizuální rozptyl (Jitter)
    # Rozptýlí body na ose Y do dvou horizontálních pruhů, aby se částice
    # nepřekrývaly a tvořily pěkný spojitý oblak.
    N_visual = len(hist['x_e'][0])
    y_jitter_e = np.random.uniform(0.2, 0.45, N_visual)
    y_jitter_i = np.random.uniform(0.55, 0.8, N_visual)

    def init_pos():
        scat_e.set_offsets(np.empty((0, 2)))
        scat_i.set_offsets(np.empty((0, 2)))
        time_text_pos.set_text('')
        return scat_e, scat_i, time_text_pos

    def animate_pos(i):
        # Maskování NaN hodnot (odstranění zničených nebo neaktivních částic)
        mask_e = ~np.isnan(hist['x_e'][i])
        mask_i = ~np.isnan(hist['x_i'][i])

        # Sestavení [x, y] pole pro scatter plot
        data_e = np.column_stack((hist['x_e'][i][mask_e], y_jitter_e[mask_e]))
        data_i = np.column_stack((hist['x_i'][i][mask_i], y_jitter_i[mask_i]))

        if len(data_e) == 0: data_e = np.empty((0, 2))
        if len(data_i) == 0: data_i = np.empty((0, 2))

        scat_e.set_offsets(data_e)
        scat_i.set_offsets(data_i)
        time_text_pos.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return scat_e, scat_i, time_text_pos

    anim_pos = animation.FuncAnimation(fig5, animate_pos, init_func=init_pos, frames=len(hist['t']), interval=100,
                                       blit=True)
    animations.append(anim_pos)
    fig5.tight_layout()

    if plot_toggles.get('save_plots', False):
        print("  -> Ukládám animaci pozic částic...")
        anim_pos.save('out_animace_pozice_castic.gif', writer='pillow', fps=15)
        print("  [OK] Uloženo: out_animace_pozice_castic.gif")


def _get_velocity_limits(hist: Dict[str, Any]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Analyzuje celou časovou historii rychlostí a nalezne globální extrémy.

    Tento krok je zcela nezbytný pro vytvoření smysluplných histogramů rychlostí.
    Kdybychom nechali osu X u histogramu "dýchat" s každým snímkem (automatický
    rescaling), vizuálně bychom zcela ztratili informaci o tom, že se rychlost
    iontů masivně zvětšuje. Fixací extrémů graf demonstruje skutečný fyzikální posun.
    """
    v_e_global_min, v_e_global_max = 0.0, 0.0
    v_i_global_min, v_i_global_max = 0.0, 0.0

    valid_v_e = [v[~np.isnan(v)] for v in hist['v_e']]
    valid_v_e = [v for v in valid_v_e if len(v) > 0]
    if valid_v_e:
        v_e_global_min = np.min([np.min(v) for v in valid_v_e])
        v_e_global_max = np.max([np.max(v) for v in valid_v_e])

    valid_v_i = [v[~np.isnan(v)] for v in hist['v_i']]
    valid_v_i = [v for v in valid_v_i if len(v) > 0]
    if valid_v_i:
        v_i_global_min = np.min([np.min(v) for v in valid_v_i])
        v_i_global_max = np.max([np.max(v) for v in valid_v_i])

    # Ochranný padding (rezerva okrajů) 5%
    pad_e = (v_e_global_max - v_e_global_min) * 0.05 if v_e_global_max != v_e_global_min else 1e5
    pad_i = (v_i_global_max - v_i_global_min) * 0.05 if v_i_global_max != v_i_global_min else 1e3

    return (v_e_global_min - pad_e, v_e_global_max + pad_e), (v_i_global_min - pad_i, v_i_global_max + pad_i)


def _animate_velocity_distribution(hist: Dict[str, Any], params: SimulationParams, plot_toggles: Dict[str, bool],
                                   v_e_lims: Tuple[float, float], v_i_lims: Tuple[float, float],
                                   animations: list) -> None:
    """
    Vytváří časově závislý dynamický histogram (Probability Density Function).

    Fyzikální kontext:
    ------------------
    Při výbuchu prachu (t=0) mají obě populace přesné polovinové Maxwell-Boltzmannovo
    tepelné rozdělení s určitou střední rychlostí a disperzí (závisí na teplotě v eV).
    Jak se ale oblak začne rozšiřovat, elektrostatické pole přelévá energii:
    elektrony jsou silně brzděny (histogram elektronů se přesune do nižších rychlostí),
    zatímco obří hmotnost iontů je urychlena vpřed do jednoho velmi úzkého vysokoenergetického
    svazku (iontový histogram se ostře přesune a zúží do špičky - beam creation).
    """
    if not (plot_toggles.get('show_velocity_anim', True) and 'v_e' in hist and 'v_i' in hist):
        return

    fig6, (ax_ve, ax_vi) = plt.subplots(2, 1, figsize=(10, 8))
    fig6.canvas.manager.set_window_title('Termodynamika: Rozložení rychlostí plazmatu')

    # Aplikace globálních extrémů na Osu X
    ax_ve.set_xlim(v_e_lims[0], v_e_lims[1])
    ax_vi.set_xlim(v_i_lims[0], v_i_lims[1])

    # Kreslíme čáru jako "schody" (steps-mid) typické pro histogramy
    line_ve, = ax_ve.plot([], [], lw=2, color='red', drawstyle='steps-mid', fillstyle='bottom')
    ax_ve.set_title("Rozložení rychlostí unikajících elektronů (Ambipolární brzdění)")
    ax_ve.set_xlabel("Okamžitá rychlost $v_e$ [m/s]")
    ax_ve.set_ylabel("Relativní četnost částic")
    ax_ve.grid(True, linestyle=':', alpha=0.7)

    line_vi, = ax_vi.plot([], [], lw=2, color='blue', drawstyle='steps-mid')
    ax_vi.set_title("Rozložení rychlostí expandujících iontů (Ambipolární urychlování)")
    ax_vi.set_xlabel("Okamžitá rychlost $v_i$ [m/s]")
    ax_vi.set_ylabel("Relativní četnost částic")
    ax_vi.grid(True, linestyle=':', alpha=0.7)

    time_text_fig6 = ax_ve.text(0.02, 0.85, '', transform=ax_ve.transAxes, fontsize=11,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))

    def init_vel():
        line_ve.set_data([], [])
        line_vi.set_data([], [])
        time_text_fig6.set_text('')
        return line_ve, line_vi, time_text_fig6

    def animate_vel(i):
        # Vytváříme histogram on-the-fly z hrubých částicových dat v daném snímku
        v_e_curr = hist['v_e'][i]
        v_e_curr = v_e_curr[~np.isnan(v_e_curr)]
        if len(v_e_curr) > 1:
            counts_e, bins_e = np.histogram(v_e_curr, bins=50, range=v_e_lims, density=False)
            centers_e = (bins_e[:-1] + bins_e[1:]) / 2.0
            line_ve.set_data(centers_e, counts_e)
            ax_ve.set_ylim(0, max(10, counts_e.max() * 1.15))

        v_i_curr = hist['v_i'][i]
        v_i_curr = v_i_curr[~np.isnan(v_i_curr)]
        if len(v_i_curr) > 1:
            counts_i, bins_i = np.histogram(v_i_curr, bins=50, range=v_i_lims, density=False)
            centers_i = (bins_i[:-1] + bins_i[1:]) / 2.0
            line_vi.set_data(centers_i, counts_i)
            ax_vi.set_ylim(0, max(10, counts_i.max() * 1.15))

        time_text_fig6.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return line_ve, line_vi, time_text_fig6

    # Zde blit=False ! Osa Y se v průběhu animace dynamicky mění,
    # což nelze efektivně překreslovat pomocí techniky blitting.
    anim_vel = animation.FuncAnimation(fig6, animate_vel, init_func=init_vel, frames=len(hist['t']), interval=100,
                                       blit=False)
    animations.append(anim_vel)
    fig6.tight_layout()

    if plot_toggles.get('save_plots', False):
        print("  -> Ukládám animaci rozložení rychlostí (s pevnou osou X)...")
        anim_vel.save('out_animace_rychlosti.gif', writer='pillow', fps=15)
        print("  [OK] Uloženo: out_animace_rychlosti.gif")


def _animate_phase_space(hist: Dict[str, Any], params: SimulationParams, plot_toggles: Dict[str, bool],
                         v_e_lims: Tuple[float, float], v_i_lims: Tuple[float, float], animations: list) -> None:
    """
    Vizualizuje strukturu fázového prostoru plazmatu (Závislost pozice na rychlosti).

    Fyzikální kontext:
    ------------------
    Fázový prostor je fundamentálním nástrojem moderní fyziky plazmatu vycházející
    z kinetické teorie (Vlasovovy rovnice). Pro každou částici v grafu určujeme
    bod o souřadnicích (x, v). Liouvilleův teorém říká, že celkový objem plazmatu
    ve fázovém prostoru se zachovává. Tření, kolize a turbulence plazmatu deformují
    tuto strukturu do složitých fraktálních tvarů.
    Tento graf odhaluje mnohem víc než pouhé "kde to je" a "jak rychle to letí",
    umí např. identifikovat vznikající fázové díry (electron holes), dvojsvazkové
    nestability (two-stream instability) či uvíznuté "trapped" částice.
    """
    if not (plot_toggles.get('show_phase_space_anim', False) and 'v_e' in hist and 'v_i' in hist):
        return

    fig7, (ax_ps_e, ax_ps_i) = plt.subplots(2, 1, figsize=(10, 9))
    fig7.canvas.manager.set_window_title('PIC Analýza: Fázový prostor (x-v)')

    # Graf Fázového prostoru elektronů
    scat_ps_e = ax_ps_e.scatter([], [], s=1, color='red', alpha=0.3, edgecolors='none')
    ax_ps_e.set_title("Fázový prostor elektronů (x, v_e)")
    ax_ps_e.set_xlabel("Vzdálenost x [m]")
    ax_ps_e.set_ylabel("Rychlost $v_e$ [m/s]")
    ax_ps_e.set_xlim(0, params.L_domain)
    ax_ps_e.set_ylim(v_e_lims[0], v_e_lims[1])
    ax_ps_e.axvline(x=params.x_antenna, color='black', linestyle='--', alpha=0.5)
    ax_ps_e.grid(True, linestyle=':', alpha=0.5)

    # Graf Fázového prostoru iontů
    scat_ps_i = ax_ps_i.scatter([], [], s=1, color='blue', alpha=0.3, edgecolors='none')
    ax_ps_i.set_title("Fázový prostor iontů (x, v_i)")
    ax_ps_i.set_xlabel("Vzdálenost x [m]")
    ax_ps_i.set_ylabel("Rychlost $v_i$ [m/s]")
    ax_ps_i.set_xlim(0, params.L_domain)
    ax_ps_i.set_ylim(v_i_lims[0], v_i_lims[1])
    ax_ps_i.axvline(x=params.x_antenna, color='black', linestyle='--', alpha=0.5)
    ax_ps_i.grid(True, linestyle=':', alpha=0.5)

    time_text_fig7 = ax_ps_e.text(0.02, 0.85, '', transform=ax_ps_e.transAxes, fontsize=11,
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9,
                                            edgecolor='gray'))

    def init_ps():
        scat_ps_e.set_offsets(np.empty((0, 2)))
        scat_ps_i.set_offsets(np.empty((0, 2)))
        time_text_fig7.set_text('')
        return scat_ps_e, scat_ps_i, time_text_fig7

    def animate_ps(i):
        # Maskování neplatných prvků napříč oběma osami nezávisle
        mask_e = ~np.isnan(hist['x_e'][i])
        mask_i = ~np.isnan(hist['x_i'][i])

        # Sestavení N x 2 matice bodů pro Scatter()
        data_e = np.column_stack((hist['x_e'][i][mask_e], hist['v_e'][i][mask_e]))
        data_i = np.column_stack((hist['x_i'][i][mask_i], hist['v_i'][i][mask_i]))

        if len(data_e) == 0: data_e = np.empty((0, 2))
        if len(data_i) == 0: data_i = np.empty((0, 2))

        scat_ps_e.set_offsets(data_e)
        scat_ps_i.set_offsets(data_i)
        time_text_fig7.set_text(f"Čas: {hist['t'][i] * 1e6:.2f} µs")
        return scat_ps_e, scat_ps_i, time_text_fig7

    anim_ps = animation.FuncAnimation(fig7, animate_ps, init_func=init_ps, frames=len(hist['t']), interval=100,
                                      blit=True)
    animations.append(anim_ps)
    fig7.tight_layout()

    if plot_toggles.get('save_plots', False):
        print("  -> Ukládám animaci fázového prostoru...")
        anim_ps.save('out_animace_fazovy_prostor.gif', writer='pillow', fps=15)
        print("  [OK] Uloženo: out_animace_fazovy_prostor.gif")


def _export_data_csv(results: Dict[str, Any], params: SimulationParams, plot_toggles: Dict[str, bool]) -> None:
    """
    Bezpečná metoda pro extrakci surových numerických hodnot měřicích obvodů.
    Uloží simulované časové řady z přístroje do standardizovaného formátu CSV
    (Comma Separated Values) pro externí zpracování a porovnání s experimentálními
    vzorky z kalibrační komory.
    """
    if not plot_toggles.get('export_data_csv', False):
        return

    print("  -> Exportuji makroskopická data do CSV formátu...")
    try:
        # Sestavení sloupcové matice
        export_matrix = np.column_stack((
            params.time_array,
            results['smooth_induced'],
            results['smooth_collected'],
            results['voltage_ant']
        ))

        header = "Cas_s,Indukovany_proud_A,Nasbirany_proud_A,Napeti_antena_V"
        np.savetxt("out_vysledky_simulace.csv", export_matrix, delimiter=",", header=header, comments="")
        print("  [OK] Data úspěšně uložena do: out_vysledky_simulace.csv")
    except Exception as e:
        print(f"  [CHYBA] Nepodařilo se exportovat CSV. Detail systémové výjimky: {e}")


def plot_simulation_results(results: Dict[str, Any], params: SimulationParams,
                            plot_toggles: Optional[Dict[str, bool]] = None) -> None:
    """
    KRYCÍ (WRAPPER) METODA PRO SPOUŠTĚNÍ VŠECH VIZUALIZAČNÍCH BLOKŮ

    Základní API volání pro tento modul. Správce provede sekvenční delegaci
    vizualizačních požadavků na specializované pod-funkce. Minimalizuje tak
    zatížení hlavní paměti a zajišťuje maximální přehlednost.

    Parametry:
    ----------
    results : dict
        Kompletní historie a vyhlazené signály (odpověď metody `run()` třídy `DustImpactSimulation`).
    params : SimulationParams
        Datová struktura obsahující konstanty a kalibraci simulace.
    plot_toggles : dict, volitelné
        Slovník boolovských hodnot spravující zapnutí/vypnutí jednotlivých funkcí.
    """
    # Pokud není specifikováno, systém povolí veškerou generovanou grafiku
    if plot_toggles is None:
        plot_toggles = {
            'show_currents': True,
            'show_fields_anim': True,
            'show_weighting_field': True,
            'show_particles_anim': True,
            'show_velocity_anim': True,
            'show_phase_space_anim': True,
            'save_plots': False,
            'export_data_csv': False
        }

    print("=====================================================")
    print("Spouštím vědecký vizualizační post-processing...")
    print("=====================================================")

    # Životně důležitá struktura udržující ukazatele na běžící FuncAnimation.
    animations = []
    hist = results['history']

    print("  -> Analyzuji distribuci historie dat a předpočítávám kinetické okraje...")
    v_e_lims, v_i_lims = _get_velocity_limits(hist)

    # Delegace na izolované podsystémy (Clean Code Pattern)
    _plot_currents_and_voltage(results, params, plot_toggles)
    _animate_macroscopic_fields(hist, params, plot_toggles, animations)
    _plot_weighting_field(params, plot_toggles)
    _animate_particles(hist, params, plot_toggles, animations)
    _animate_velocity_distribution(hist, params, plot_toggles, v_e_lims, v_i_lims, animations)
    _animate_phase_space(hist, params, plot_toggles, v_e_lims, v_i_lims, animations)
    _export_data_csv(results, params, plot_toggles)

    print("=====================================================")
    print("Post-processing ukončen. Všechny vizualizace byly úspěšně připraveny.")

    # Blokující volání Matplotlibu - udrží okna otevřená a běžící
    if animations:
        print(
            f"Spouštím okna s grafy (Počet aktivních interaktivních animací: {len(animations)}). Zavřete okna pro ukončení programu.")
        plt.show()
    else:
        print("Nebyly aktivovány žádné animace. Zobrazuji blokující statické grafy.")
        plt.show()