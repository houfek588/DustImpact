# Validační výpočty

Skripty v této složce porovnávají výsledky 3D PIC simulátoru s analytickými řešeními. Spouštějí se
z kořene repozitáře. Všechny generované soubory patří do `outputs/`.

| Skript | Co dělá |
|---|---|
| `debye_profile.py` | Profil potenciálu kolem nabité koule zprůměrovaný přes kulové slupky a čas. Porovná ho s Boltzmannovou rovnováhou ve stejném uzavřeném boxu na stejné mřížce a s otevřeným Debye-Hückelovým profilem. Ionty v referenci jsou nehybné, nebo boltzmannovské, pokud běh trvá déle než 2 iontové plazmové periody (`--ions`). |
| `debye_plots.py` | Graf potenciálu na ose x v čase T a animace všech snímků, obojí s průměrem přes kulové slupky a Debye-Hückelovým profilem. |
| `escape_check.py` | Únik testovacích částic oblaku od koule s předpětím. Porovnává podíl uniklých částic s analytickým vzorcem $P = \mathrm{erfc}(\sqrt{x}) + 2\sqrt{x/\pi}\,e^{-x}$. Simulaci spouští sám, protože výstup `main.py` neukládá, kde částice skončily. |
| `staircase_convergence.py` | Chyba schodovité (voxelizované) koule na různých mřížkách, bez částic. Ukazuje, jakou odchylku od teorie způsobí samotná mřížka. |

## Doporučené výpočty

| # | Konfigurace | Spuštění | Analytické očekávání |
|---|---|---|---|
| 1 | `inputs/config_validation_escape_p5.json` | `python scripts/validation/escape_check.py inputs/config_validation_escape_p5.json --steps 1000` | uniklé elektrony 19,0 % |
| 2 | `inputs/config_validation_escape_m5.json` | `python scripts/validation/escape_check.py inputs/config_validation_escape_m5.json` | uniklé ionty ≈ 59 % (20 000 kroků) |
| 3 | `inputs/config_validation_escape_0.json` | `python scripts/validation/escape_check.py inputs/config_validation_escape_0.json --steps 1000` | uniklé všechny částice |
| 4 | `inputs/config_analytical_debye2.json` | `python -u main.py --config inputs/config_analytical_debye2.json --dim 3 --no-visualize` | rovnováha v boxu; DH s $\lambda_{De}$ = 1,05 m |
| 5 | `inputs/config_validation_debye_ions.json` | `python -u main.py --config inputs/config_validation_debye_ions.json --dim 3 --no-visualize` | $\lambda_{\text{eff}} = \lambda_{De}/\sqrt{2}$ = 0,74 m |
| 6 | `inputs/config_shen2023_fig6.json` | `python -u main.py --config inputs/config_shen2023_fig6.json --dim 3 --no-visualize` | tvar monopólového signálu z Obr. 6 v Shen et al. (2023) |

### Série Obr. 2 (Shen et al. 2023)

Devět konfigurací `inputs/config_shen2023_fig2_<předpětí>_<úhel>.json`: předpětí `0V`, `p5V`, `m5V` na tělese
i všech anténách a dopad na rovník pod úhlem `10deg`, `30deg`, `45deg` od ANT3 směrem k ANT4. Antény
ANT1–ANT4 leží na 45°, 135°, 225° a 315°. Ostatní parametry jsou stejné jako u Obr. 6: plovoucí sonda,
tenké antény, Q = 7,8·10⁻¹⁴ C, ionty 9,8 km/s v kuželu 45°, izotropní elektrony a záznam 160 µs.

```bash
python -u main.py --config inputs/config_shen2023_fig2_0V_10deg.json --dim 3 --no-visualize
```

Očekávané trendy z článku:
- dipól `V_ANT,3 - V_ANT,4` je nejvýraznější u 10°, slábne u 30° a u 45° je téměř nulový. Jeho kladná
  část trvá asi 30 µs (L/v_i);
- monopóly ANT1 a ANT2 jsou si podobné, protože obě antény leží daleko od místa dopadu;
- při +5 V je záporná elektronová špička monopólů menší než při 0 V a kladná část zůstává;
- při −5 V převažuje záporný signál, protože víc elektronů uniká a sonda sebere část iontů.

Vyhodnocení běhů 4 a 5:

```bash
python scripts/validation/debye_profile.py --config inputs/config_analytical_debye2.json \
    --results outputs/out_analytical_debye2.h5 --plot outputs/debye2_profil.png
python scripts/validation/debye_plots.py --config inputs/config_analytical_debye2.json \
    --results outputs/out_analytical_debye2.h5 --out-prefix outputs/debye2
```

Běh 6 vykreslí monopóly a dipóly přímo `main.py`, pokud se vynechá `--no-visualize`. Jinak je
vykreslí následné spuštění s `--plot-only`.

Poznámky:
- Výsledný soubor vyhodnocení musí pocházet ze stejné konfigurace, protože geometrie se znovu sestaví
  z konfiguračního souboru.
- U `escape_check.py` je součet „escaped + still flying“ horní mez konečného podílu uniklých částic.
  Pomalé částice se ještě mohou vrátit, proto je potřeba dost kroků.
- Pro srovnání se SPIS se hodí běh 5: SPIS má pohyblivé ionty a otevřenou hranici, takže by měl dát
  otevřený Debye-Hückelův profil s $\lambda \approx 0{,}74$ m bez kladného posunu daleko od koule.
