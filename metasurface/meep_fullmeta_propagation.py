import meep as mp
mp.verbosity(1)

import numpy as np
import matplotlib.pyplot as plt

resolution = 15 #[px/um]

#all dimensions in [um]
lda0 = 1.55
freq0 = 1 / lda0
wvl_min = 1.5
wvl_max = 1.6
wvl_cen = lda0
fmin = 1/wvl_max
fmax = 1/wvl_min
fcen = 1/wvl_cen
df = fmax-fmin
fwidth = df
nfreq = 1

dpml = 0.5 #PML thickness
dsub = 0.2*lda0 #substrate thickness
#dpad = dsub #padding between cell and PML

n_si = 3.45 #silicon refractive index
si = mp.Medium(index = n_si)
n_sio2 = 1.45
sio2 = mp.Medium(index = n_sio2)
n_air = 1.0 #air refractive index
air = mp.Medium(index = n_air)

P = 0.7 #unit cell pitch [um]
unit_cell_pitch = P
h = 0.9 #pillar height [um]

#propgation distances
d0 = 535 #[um]
d1 = 415 #[um]
d2 = d0 #[um]

Nx = 1025 #pixels per dimension
Ny = Nx
nx = (Nx - 1) // 2 #indices without axes (= #512)
ny = (Ny - 1) // 2
dof = 2*((nx + 1)**2) #number of degrees of freedom per metasurface
S = Nx * Ny

size_x = unit_cell_pitch * Nx #actual physical size
size_y = unit_cell_pitch * Ny

xs = np.arange(-Nx//2+1, Nx//2+1) * (size_x / Nx)
ys = np.arange(-Ny//2+1, Ny//2+1) * (size_y / Ny)
print(xs[0],xs[-1])
X, Y = np.meshgrid(xs, ys)
rho = np.sqrt(X**2 + Y**2)


####################################################
#LOAD PHASE MASK
#___________________________________________________
#load the phase mask data
phase_mask_filename = "results_sym/optimized_phase_mask_1.npy"
phase_mask = np.load(phase_mask_filename)

#sim_domain_pixel_size = #full simulation domain (full axis)
sim_domain_size = 140 #[um]

max_sim_radius = 60

X_quadrant = X[nx:, ny:]
xs_quadrant = xs[nx:]
xs_sim = xs[np.abs(xs) <= sim_domain_size/2]
xs_sim_quadrant = xs_quadrant[np.abs(xs_quadrant) <= sim_domain_size/2]

Y_quadrant = Y[nx:, ny:]
ys_quadrant = ys[ny:]
ys_sim = ys[np.abs(ys) <= sim_domain_size/2]
ys_sim_quadrant = ys_quadrant[np.abs(ys_quadrant) <= sim_domain_size/2]

X_sim, Y_sim = np.meshgrid(xs_sim, ys_sim)

phase_mask_quadrant = phase_mask[nx:, ny:]

#import the input field (just before MS1)
input_field = np.load("results_sym/input_field.npy")

mask = (np.abs(X) <= sim_domain_size/2) & (np.abs(Y) <= sim_domain_size/2)

eta_0 = 1 #due to meep's internal normalization
Ex_field_after_MS1 = input_field * np.exp(1j*phase_mask) / np.max(np.abs(input_field))

Ex_field_after_MS1_sim = Ex_field_after_MS1[mask].reshape((len(xs_sim), len(ys_sim)))
Hy_field_after_MS1_sim = Ex_field_after_MS1_sim / eta_0

# [markdown]
# for the equivalence principle, if we want to use the field after MS1
# as an actual source, we need to define proper magnetic and electric currents
# we assume the input electric field is polarized along the x-axis
# and the magnetic field is polarized along the y-axis.
# Specifying both electric and magnetic fields is necessary
# to obtain a directional source. 

#define the source field dataset
source_position = mp.Vector3(0,0,-0.1)

#the surface currents due to the equivalance principle are:
#Js_x = -Hy_field_after_MS1
#Ms_y = -Ex_field_after_MS1

sources = [
    mp.Source(#electric current
        src=mp.GaussianSource(frequency=freq0, fwidth=fwidth),
        component=mp.Ex,
        center=source_position,
        size=mp.Vector3(sim_domain_size, sim_domain_size, 0),
        amplitude=-1,
        amp_data=Hy_field_after_MS1_sim[:, :, np.newaxis]
    ),

    mp.Source(#magnetic current
        src=mp.GaussianSource(frequency=freq0, fwidth=fwidth),
        component=mp.Hy,
        center=source_position,
        size=mp.Vector3(sim_domain_size, sim_domain_size, 0),
        amplitude=-1,
        amp_data=Ex_field_after_MS1_sim[:, :, np.newaxis]
    ),
]

pml_layers = [mp.PML(dpml)]

Lx = sim_domain_size + 2*dpml
Ly = sim_domain_size + 2*dpml
Lz = 2*dpml + 0.4

cell_size = mp.Vector3(Lx, Ly, Lz)

sim = mp.Simulation(
    resolution=resolution,
    cell_size = cell_size,
    boundary_layers = pml_layers,
    default_material = air,
    sources = sources,
    symmetries = [mp.Mirror(0, phase=-1), mp.Mirror(1, phase=+1)]
)

#far field points for projection monitor
ff_resolution = 5
ff_plane_size = 30 #[um]

xs_far = np.linspace(-ff_plane_size/2, ff_plane_size/2, ff_resolution * ff_plane_size)
ys_far = np.linspace(-ff_plane_size/2, ff_plane_size/2, ff_resolution * ff_plane_size)

X_far, Y_far = np.meshgrid(xs_far, ys_far)

projection_distance = d1 - np.abs(source_position.z)

near_region_monitor = mp.Near2FarRegion(
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(sim_domain_size, sim_domain_size, 0),
)

n2f_obj = sim.add_near2far([freq0], near_region_monitor)

#flux_mon = sim.add_flux(
#        [fcen],
#        mp.FluxRegion(center=mp.Vector3(0,0,0), size=mp.Vector3(sim_domain_size, sim_domain_size, 0))
#    )

fig, ax = plt.subplots()
plot_xz_plane = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(Lx, 0, Lz))
sim.plot2D(ax=ax,output_plane=plot_xz_plane)
ax.set_aspect('auto')

obs_point = mp.Vector3(0,0,0)
stop_cond = mp.stop_when_fields_decayed(50, mp.Ez, obs_point, 1e-9)

sim.run(until_after_sources=stop_cond)

far_plane = mp.Volume(
    center=mp.Vector3(0, 0, projection_distance),
    size=mp.Vector3(ff_plane_size, ff_plane_size, 0)
)

ff_data = sim.get_farfields(n2f_obj, ff_resolution, where=far_plane)

ff_Ex = ff_data['Ex']
np.save("ff_Ex.npy", ff_Ex)

fig, ax = plt.subplots()
im = ax.pcolormesh(X_far, Y_far, np.abs(ff_Ex), cmap='viridis', shading='auto')
ax.set_xlabel('X [um]')
ax.set_ylabel('Y [um]')
fig.colorbar(im, ax=ax)
fig.savefig("magnitude_field_before_MS2.pdf")

#make a line plot that is a slice along the y=0 axis
fig, az = plt.subplots()
im = az.plot(xs_far, np.abs(ff_Ex[ff_resolution * ff_plane_size//2, :]))
az.set_xlabel('X [um]')
az.set_ylabel('Magnitude of E-field')
fig.savefig("magnitude_field_before_MS2_slice.pdf")


