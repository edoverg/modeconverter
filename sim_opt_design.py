#this file serves to simulate the optimized design parameters
#it imports data from the optimization process and runs a meep simulation to verify the results
import faulthandler
faulthandler.enable()
from typing import List, NamedTuple, Tuple
from autograd import numpy as npa, tensor_jacobian_product, grad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import meep as mp
import meep.adjoint as mpa
import nlopt
import numpy as np
import os
import xarray as xr
from PIL import Image
import pickle

if not os.path.exists("sim_opt_design_results"):
    os.makedirs("sim_opt_design_results")
#import data from previous savez file
try:
    with open("results/optimal_design.pkl", "rb") as f:
        data = pickle.load(f)
        RESOLUTION=data['RESOLUTION']
        MAX_RUN_TIME = data['MAX_RUN_TIME']
        WAVELENGTH_MIN_UM = data['WAVELENGTH_MIN_UM']
        WAVELENGTH_MAX_UM = data['WAVELENGTH_MAX_UM']
        DESIGN_REGION_CENTER = data['DESIGN_REGION_CENTER']
        SILICON = data['SILICON']
        SILICON_DIOXIDE = data['SILICON_DIOXIDE']
        AIR = data['AIR']
        cell_um = data['cell_um']
        SUBSTRATE_THICKNESS = data['SUBSTRATE_THICKNESS']
        SUBSTRATE_SIZE = data['SUBSTRATE_SIZE']
        SUBSTRATE_CENTER = data['SUBSTRATE_CENTER']
        physical_domains_size = data['physical_domains_size']
        frequency_min = data['frequency_min']
        frequency_max = data['frequency_max']
        frequency_center = data['frequency_center']
        frequency_width = data['frequency_width']
        frequencies = data['frequencies']
        num_wavelength = data['num_wavelengths']
        SOURCE_CENTER = data['SOURCE_CENTER']
        SOURCE_SIZE = data['SOURCE_SIZE']
        NEAR_REGION_MONITOR_CENTER = data['NEAR_REGION_MONITOR_CENTER']
        NEAR_REGION_MONITOR_SIZE = data['NEAR_REGION_MONITOR_SIZE']
        FF_MONITOR_CENTER = data['FF_MONITOR_CENTER']
        FF_MONITOR_SIZE = data['FF_MONITOR_SIZE']
        xs = data['xs']
        ys = data['ys']
        ff_points = data['ff_points']
        DESIGN_WAVELENGTHS_UM=data['DESIGN_WAVELENGTHS_UM']
        BUFFER=data['BUFFER']
        PML_UM=data['PML_UM']
        DESIGN_REGION_UM=data['DESIGN_REGION_UM']
        DESIGN_REGION_RESOLUTION=data['DESIGN_REGION_RESOLUTION']
        NX_DESIGN_GRID=data['NX_DESIGN_GRID']
        NY_DESIGN_GRID=data['NY_DESIGN_GRID']
        MIN_LENGTH_UM=data['MIN_LENGTH_UM']
        sigmoid_biases=data['sigmoid_biases']
        max_eval=data['max_eval']
        objfunc_history=data['objfunc_history']
        epivar_history=data['epivar_history']
        epigraph_variable=data['epigraph_variable']
        unmapped_design_weights=data['unmapped_design_weights']
        optimal_design_weights=data['optimal_design_weights']
except FileNotFoundError:
    raise FileNotFoundError("Optimal design data not found.")

print(optimal_design_weights)
pml_layers = [mp.PML(PML_UM)]

view_2D_plane_xz = mp.Volume(
    center=mp.Vector3(0,0,0), 
    size=mp.Vector3(cell_um.x,0,cell_um.z)
)

view_2D_plane_xy = mp.Volume(
    center=mp.Vector3(0,0,DESIGN_REGION_CENTER.z/2), 
    size=mp.Vector3(cell_um.x,cell_um.y,0)
)

matgrid = mp.MaterialGrid(
    grid_size = mp.Vector3(NX_DESIGN_GRID, NY_DESIGN_GRID,0),
    medium1 = AIR,
    medium2 = SILICON,
    weights = optimal_design_weights.reshape((NX_DESIGN_GRID, NY_DESIGN_GRID))
)

matgrid_region = mpa.DesignRegion(
    matgrid,
    volume=mp.Volume(
        center=DESIGN_REGION_CENTER, 
        size=DESIGN_REGION_UM
    )
)

matgrid_block = mp.Block(
    center=matgrid_region.center,
    size=matgrid_region.size,
    material=matgrid,
)

substrate_block = mp.Block(
    center = SUBSTRATE_CENTER,
    size = SUBSTRATE_SIZE,
    material = SILICON_DIOXIDE,
)

geometry = [
    matgrid_block,
    substrate_block,
]

sources = [
    mp.Source(
        src=mp.GaussianSource(frequency_center, fwidth=frequency_width),
        component=mp.Ex,
        size=SOURCE_SIZE,
        center=SOURCE_CENTER,
    )
]

sim = mp.Simulation(
    resolution=RESOLUTION,
    default_material=AIR,
    cell_size=cell_um,
    sources=sources,
    geometry=geometry,
    boundary_layers=pml_layers,
    k_point=mp.Vector3(),
)

plt.figure()
ax = plt.gca()
sim.plot2D(output_plane=view_2D_plane_xz, ax=ax)
plt.savefig(f"sim_opt_design_results/simulation_setup.pdf")
plt.close()

NearRegions = [
    mp.Near2FarRegion(
        center=NEAR_REGION_MONITOR_CENTER,
        size=NEAR_REGION_MONITOR_SIZE,
        weight=+1,
    )
]

FarFields = mpa.Near2FarFields(
    sim=sim,
    Near2FarRegions=NearRegions,
    far_pts=ff_points,
)
obj_list = [FarFields]

def dummy_J1(ff):
    return [1]*num_wavelength

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[dummy_J1],
    objective_arguments=[FarFields],
    design_regions=[matgrid_region],
    frequencies=frequencies,
    maximum_run_time=MAX_RUN_TIME,
)

#make a plot of the optimization setup
plt.figure(figsize=(12,6))
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
opt.plot2D(True, output_plane=view_2D_plane_xz, ax=ax1)

opt.plot2D(True, output_plane=view_2D_plane_xy, ax=ax2)

plt.savefig(f"sim_opt_design_results/optimization_setup.pdf")

plt.close()


print(type(optimal_design_weights), optimal_design_weights.shape)
dummy = opt(
    [
        optimal_design_weights.flatten()
    ], 
    need_gradient=False,
)

#note that get_objective_arguments returns 
# a LIST of the arguments passed to the obj. fun.
#we select the first list element, since we ony have one obj. fun.
solved_ff = opt.get_objective_arguments()[0]

def intensity_from_farfields(FarFields) -> np.ndarray:
    '''Computes the intensity pattern at the monitor location from the simulated far-fields

    Args:
        FarFields: the simulated far-field patterns at the farfield 
        monitor location, for each wavelength
    Returns:
        A 2D-array [Nx*Ny,num_wavelengths] of intensities, for each wavelength,
          at the monitor location.
    '''
    intensity_x = npa.abs(FarFields[:,:,0]) ** 2

    return intensity_x

solved_intensity = intensity_from_farfields(solved_ff)

#make a plot of the ff intensities (at the center frequency)
array_to_plot = solved_intensity[:,1].reshape(NX_DESIGN_GRID, NY_DESIGN_GRID)
plt.figure()
plt.imshow(array_to_plot, origin='lower', extent=(-cell_um.x/2,cell_um.x/2,-cell_um.y/2,cell_um.y/2))
plt.xlabel("x (um)")
plt.ylabel("y (um)")
plt.title("Simulated far-field intensity at center frequency")
plt.colorbar(label="Intensity (a.u.)")
plt.savefig(f"sim_opt_design_results/simulated_farfield_intensity.pdf")
plt.close()

