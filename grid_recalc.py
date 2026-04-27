import pyvista as pv
import numpy as np


def prepocet_spis_do_numpy(vtk_soubor, rozliseni=(200, 200, 200)):
    print(f"Načítám nestrukturovaná data ze SPISu: {vtk_soubor}...")
    # 1. Načtení VTK souboru (nepravidelná tetraedrická síť)
    spis_mesh = pv.read(vtk_soubor)

    # Zjištění rozměrů simulačního boxu (min a max pro x, y, z)
    bounds = spis_mesh.bounds
    print(f"Hranice boxu (x_min, x_max, y_min...): {bounds}")

    # 2. Vytvoření prázdné pravidelné mřížky (ImageData / UniformGrid)
    print(f"Vytvářím pravidelnou mřížku s rozlišením {rozliseni}...")

    # Výpočet kroku mřížky (dx, dy, dz)
    spacing = (
        (bounds[1] - bounds[0]) / (rozliseni[0] - 1),
        (bounds[3] - bounds[2]) / (rozliseni[1] - 1),
        (bounds[5] - bounds[4]) / (rozliseni[2] - 1)
    )

    # Vytvoření samotného gridu
    grid = pv.ImageData(
        dimensions=rozliseni,
        spacing=spacing,
        origin=(bounds[0], bounds[2], bounds[4])
    )

    # 3. INTERPOLACE - To nejdůležitější kouzlo
    print("Interpoluji data ze SPISu na pravidelnou mřížku (tohle může chvíli trvat)...")
    # Metoda 'sample' vezme hodnoty ze spis_mesh a napasuje je na náš nový 'grid'
    resampled_grid = grid.sample(spis_mesh)

    # 4. Extrakce do čistého 3D NumPy pole
    # PyVista ukládá point_data jako 1D pole, musíme ho přeskládat (reshape).
    # Pozor: VTK používá Fortran ('F') pořadí pro indexování v paměti!

    # Předpokládejme, že se vaše proměnná jmenuje 'Potential' (ověřte si název ve SPISu)
    if 'plasma pot at t = 1.0 s' in resampled_grid.point_data:
        potencial_1d = resampled_grid.point_data['plasma pot at t = 1.0 s']
        potencial_3d_numpy = potencial_1d.reshape(rozliseni, order='F')
        print("Hotovo! 3D pole potenciálu je připraveno.")
        return potencial_3d_numpy, spacing, bounds
    else:
        print("Chyba: Proměnná 'Potential' nenalezena. Dostupná data:", resampled_grid.point_data.keys())
        return None, None, None


# Použití
vtk_file = "pot1t.vtk"
# U Solar Orbiteru volte vyšší rozlišení, abyste neztratili ty tenké antény!
potencial_pole, krok_mrizky, hranice = prepocet_spis_do_numpy(vtk_file, rozliseni=(300, 300, 300))