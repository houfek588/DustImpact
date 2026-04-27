import netCDF4 as nc
import numpy as np
import pyvista as pv

nc_path = ("C:/Projects/Phd/SPIS/MyProj/test3/DefaultProject.spis5/DefaultStudy/Simulations/Run1/OutputFolder/DataFieldExtracted/"
           "log10_of_final_total_density_-_step0.nc")
var_name = "rho"     # <-- your variable
x_name, y_name, z_name = "x", "y", "z"  # <-- your coord names
t_name = "time"      # <-- optional

# Open the NetCDF file in read mode
dataset = nc.Dataset(nc_path, mode="r")

# List all variables in the file
print("Variables in file:", list(dataset.variables.keys()))

# Access a specific variable (example: 'temperature')
if 'dataArray' in dataset.variables:
    temperature = dataset.variables['dataArray'][:]  # Load all data
    print("dataArray shape:", temperature.shape)
    print("dataArray sample:", temperature[:5])

if 'meshElmentId' in dataset.variables:
    temperature = dataset.variables['meshElmentId'][:]  # Load all data
    print("meshElmentId shape:", temperature.shape)
    print("meshElmentId sample:", temperature[:5])
# with Dataset(nc_path) as ds:
#     print(ds)
#     print(ds.variables["dataArray"].size)
#     print(ds.variables["dataArray"].dimensions)
#
#
#     x = ds.variables[x_name][:]
#     y = ds.variables[y_name][:]
#     z = ds.variables[z_name][:]
#
#     # pick time index if present
#     v = ds.variables[var_name]
#     if t_name in v.dimensions:
#         # assume dims like (time, z, y, x) — change if needed
#         data3d = v[0, :, :, :]   # take first time slice
#     else:
#         data3d = v[:, :, :]      # (z, y, x)
#
# # Ensure float32 and contiguous
# data3d = np.ascontiguousarray(data3d.astype(np.float32))
#
# # UniformGrid requires origin (mins) and spacing (Δ)
# dx = float(np.diff(x).mean())
# dy = float(np.diff(y).mean())
# dz = float(np.diff(z).mean())
# origin = (float(x.min()), float(y.min()), float(z.min()))
# spacing = (dx, dy, dz)
#
# nx, ny, nz = len(x), len(y), len(z)
#
# grid = pv.UniformGrid()
# grid.dimensions = (nx, ny, nz)     # VTK uses (nx, ny, nz)
# grid.origin = origin
# grid.spacing = spacing
#
# # VTK expects Fortran ordering for structured grids
# grid[var_name] = data3d.transpose(2,1,0).ravel(order="F")  # (x,y,z) then flatten
#
# # Quick viz (volume or isosurface)
p = pv.Plotter()
p.add_mesh(temperature, scalars=var_name)   # or: grid.contour(10)
p.show()



