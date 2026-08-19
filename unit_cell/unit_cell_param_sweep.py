import meep as mp
mp.verbosity(0)

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

resolution = 50 # pixels/unit length

wvl_min = 1.5
wvl_max = 1.6
wvl_cen = 0.5*(wvl_min+wvl_max)
fmin = 1/wvl_max
fmax = 1/wvl_min
fcen = 1/wvl_cen
df = fmax-fmin
nfreq = 20

dpml = 1.0 #PML thickness
dsub = 1.5 + wvl_cen #substrate thickness
dpad = dsub #padding between cell and PML

k_point = mp.Vector3(0,0,0) #used for PBC

n_si = 3.45 #silicon refractive index
si = mp.Medium(index=n_si)
n_sio2 = 1.45
sio2 = mp.Medium(index=n_sio2)
n_air = 1.0 #air refractive index
air = mp.Medium(index=n_air)

P = 0.7 #unit cell pitch
h = 0.9 #pillar height

def unitCell_paramsweep(D):
    print("Rank {}: Running simulation for diameter {}".format(rank, D))
    sim_x_size = dpml + dsub + dpad + dpml
    sim_z_size = P
    sim_y_size = P

    cell_size = mp.Vector3(sim_x_size, sim_y_size, sim_z_size)

    source_position = mp.Vector3(-wvl_cen,0,0) #source position in the middle of the substrate
    source = mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component = mp.Ez,
        center = source_position,
        size = mp.Vector3(0, sim_y_size, sim_z_size),
    )

    pml_layers = [mp.PML(dpml,direction=mp.X)]

    monitor_position = mp.Vector3(h + wvl_cen,0,0)
    obs_point = monitor_position
    stop_cond = mp.stop_when_fields_decayed(50, mp.Ez, obs_point, 1e-9)
    #stop_cond = 10
    sim = mp.Simulation(
        resolution = resolution,
        cell_size = cell_size,
        boundary_layers = pml_layers,
        k_point = k_point,
        default_material = sio2,
        sources = [source],
    )

    flux_mon = sim.add_flux(
        fcen, 
        df,
        nfreq, 
        mp.FluxRegion(center=monitor_position, size=mp.Vector3(0, sim_y_size, sim_z_size))
    )

    sim.run(until_after_sources=stop_cond)
    input_flux = mp.get_fluxes(flux_mon)

    sim.reset_meep()

    pillar_geo = mp.Cylinder(
        material = si,
        radius = D/2,
        height = h,
        center = mp.Vector3(h/2,0,0),
        axis=mp.Vector3(1,0,0),
    )

    substrate_geo = mp.Block(
        material = sio2,
        size = mp.Vector3(dsub+dpml, mp.inf, mp.inf),
        center = mp.Vector3(-(dsub+dpml)/2,0,0),
    )

    geometries = [substrate_geo, pillar_geo]

    sim = mp.Simulation(
        resolution = resolution,
        cell_size = cell_size,
        boundary_layers = pml_layers,
        geometry = geometries,
        k_point = k_point,
        sources = [source],
    )

    mode_mon = sim.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=monitor_position, size=mp.Vector3(0, sim_y_size, sim_z_size))
    )

    sim.run(until_after_sources=stop_cond)

    freqs = mp.get_eigenmode_freqs(mode_mon)
    res = sim.get_eigenmode_coefficients(
        mode_mon, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y
    )
    coeffs = res.alpha

    mode_wvl = [1/freqs[nf] for nf in range(nfreq)]
    mode_tran = [abs(coeffs[0,nf,0])**2 / input_flux[nf] for nf in range(nfreq)]
    mode_phase = [np.angle(coeffs[0,nf,0]) for nf in range(nfreq)]

    return mode_wvl, mode_tran, mode_phase

if rank == 0:
    diameters = np.linspace(0.1, 0.6, 24)
else:
    diameters = None
diameters = comm.bcast(diameters, root=0)
local_diameters = np.array_split(diameters, size)[rank]
local_indices = np.array_split(np.arange(len(diameters)), size)[rank]

local_mode_tran_arr = np.zeros((len(diameters), nfreq))
local_mode_phase_arr = np.zeros((len(diameters), nfreq))

for local_i, (global_i, D) in enumerate(zip(local_indices,local_diameters)):
    mode_wvl, mode_tran, mode_phase = unitCell_paramsweep(D)
    local_mode_tran_arr[global_i,:] = mode_tran
    local_mode_phase_arr[global_i,:] = mode_phase


mode_tran_arr = comm.gather(local_mode_tran_arr, root=0)
mode_phase_arr = comm.gather(local_mode_phase_arr, root=0)

if rank == 0:
    final_mode_tran_arr = np.zeros((len(diameters), nfreq))
    final_mode_phase_arr = np.zeros((len(diameters), nfreq))
    num_diameters_per_rank = len(diameters) // size
    for i in range(0, size):
        for j in range(num_diameters_per_rank):
           global_index = i * num_diameters_per_rank + j
           final_mode_tran_arr[global_index, :] = mode_tran_arr[i][global_index, :]
           final_mode_phase_arr[global_index, :] = mode_phase_arr[i][global_index, :]


    np.save("unit_cell_param_sweep_diameters.npy", diameters)
    np.save("unit_cell_param_sweep_mode_wvl.npy", mode_wvl)
    np.save("unit_cell_param_sweep_mode_tran.npy", final_mode_tran_arr)
    np.save("unit_cell_param_sweep_mode_phase.npy", final_mode_phase_arr)

    plt.figure()
    plt.subplot(1,2,1)
    plt.pcolormesh(
        mode_wvl,
        diameters,
        final_mode_tran_arr,
        cmap='hot_r',
        shading='gouraud',
        vmin=0,
        vmax=final_mode_tran_arr.max(),
    )

    plt.axis([wvl_min, wvl_max, diameters[0], diameters[-1]])
    plt.xlabel("Wavelength (um)")
    plt.xticks([t for t in np.linspace(wvl_min, wvl_max, 3)])
    plt.ylabel("Pillar Diameter (um)")
    plt.title("Transmission")
    cbar=plt.colorbar()
    cbar.set_ticks([t for t in np.arange(0, 1.2, 0.2)])

    plt.subplot(1,2,2)
    plt.pcolormesh(
        mode_wvl,
        diameters,
        final_mode_phase_arr,
        cmap='RdBu',
        shading='gouraud',
        vmin=final_mode_phase_arr.min(),
        vmax=final_mode_phase_arr.max(),
    )

    plt.axis([wvl_min, wvl_max, diameters[0], diameters[-1]])
    plt.xlabel("Wavelength (um)")
    plt.xticks([t for t in np.linspace(wvl_min, wvl_max, 3)])
    plt.ylabel("Pillar Diameter (um)")
    plt.title("phase")
    cbar=plt.colorbar()
    cbar.set_ticks([t for t in np.arange(-3, 4, 1)])


    plt.subplots_adjust(wspace=0.5)
    plt.savefig("unit_cell_param_sweep.pdf")

