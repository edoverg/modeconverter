# %% [markdown]
#  This code implements an inverse designed metasurface that is used to recreate a given intenstity
# 
#  pattern. It is based on meep and tidy3d tutorials.

# %%
import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product, grad
import nlopt
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import scipy as sp
import os

mp.verbosity(2)

#setup a results folder
if not os.path.exists("results"):
    os.makedirs("results")


# %%
######PHYSICS SETUP######
subs_index = 3.4
subs_perm = subs_index**2

Si = mp.Medium(index = subs_index)
Air = mp.Medium(index = 1.0)

wavelength = 1.55
nf = 1 #number of frequencies to test
frequencies = np.array([1 / 1.55])

#distance between PML and source/monitor
buffer = 2.5

######PML######
pml_size = 1.0
pml_layers = [mp.PML(pml_size)]


# %%
######FILTER######
minimum_length = 0.09  # minimum length scale (microns)
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)


# %%
######DESIGN REGION######
design_region_width = 4.0
design_region_length = 4.0
design_region_height = (wavelength/np.sqrt(subs_index))/2 #half-wavelength in the substrate

resolution = 5 # pixels/um
design_region_resolution = int(resolution)

#number of pixels in the design region
Nx = int(design_region_width * design_region_resolution) + 1
Ny = int(design_region_length * design_region_resolution) + 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type="U_MEAN")

design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center = mp.Vector3(0,0,design_region_height / 2),
        size = mp.Vector3(
                    design_region_width, 
                    design_region_length, 
                    design_region_height),
    ),
)

######CELL GEOMETRY######
Sx = 2 * pml_size + design_region_width
Sy = 2 * pml_size + design_region_length
Sz = 2 * pml_size + design_region_height + 4*buffer
cell_size = mp.Vector3(Sx, Sy, Sz)

substrate_thickness = Sz / 2
substrate_size = mp.Vector3(design_region_width + 2 * pml_size, 
                            design_region_length + 2 * pml_size, 
                            substrate_thickness)

substrate_center = mp.Vector3(0, 0, - substrate_thickness / 2)

substrate_block = mp.Block(
    center = substrate_center,
    size = substrate_size,
    material = Si,
)

design_region_block = mp.Block(
    center = design_region.center, 
    size = design_region.size, 
    material = design_variables,
)

geometry = [#combine geometries
    substrate_block,
    design_region_block,
]

# %%
######SOURCE######
fcen = 1 / wavelength
width = 0.2
fwidth = width * fcen
source_center = [0, 0, -3]
source_size = mp.Vector3(design_region_width, design_region_length, 0)
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]

# %%
######OBSERVATION PLANES######
#planes used for references in 2D plots
output_plane = mp.Volume(
    center=mp.Vector3(0,0,0), 
    size=mp.Vector3(Sx,0,Sz))

output_plane_xy = mp.Volume(
    center=mp.Vector3(0,0,design_region_height / 2),
    size=mp.Vector3(Sx,Sy,0)
)

# %%
######SIMULATION######
sim_run_time = 100
kpoint = mp.Vector3()
sim = mp.Simulation(
    cell_size = cell_size,
    boundary_layers = pml_layers,
    geometry = geometry,
    sources = source,
    default_material = Air,
    resolution = resolution,
)

sim.plot2D(output_plane=output_plane,show_sources=True, show_boundaries=True, show_geometry=True)
plt.savefig("results/simulation_geometry.pdf")

# %%
import xarray as xr
from PIL import Image

logo_name = "misc/peg.png"

def get_logo() -> np.ndarray:
    logo = Image.open(logo_name).convert("L") #L mode stands for grayscale
    logo = np.array(logo).astype(float)
    #logo is normalized between 0 and 1
    logo -= np.min(logo)
    logo /= np.max(logo)
    #in general, logo has any shape. At a later stage it will
    #be rescaled to fit the design region
    print("Logo shape: ", logo.shape)
    return logo

def intensity_desired_fn_logo(xs: list, ys: list, rescale: float = 0.5) -> np.ndarray:
    '''return the value of the logo as a function of the x,y coordinates, 
    with a rescale factor
    Parameters:
        xs: list of x coordinates
        ys: list of y coordinates
    Returns:
        logo_interp: 2D array of the same shape as the design region (nx,ny)
    '''
    logo_values = get_logo()
    logo_values = 1 - np.rot90(np.rot90(np.rot90(logo_values))) #rotate the logo by 90 degrees to match the orientation of the simulation
    
    nx,ny = logo_values.shape
    xs_logo = np.linspace(rescale*min(xs), rescale*max(xs), nx)
    ys_logo = np.linspace(rescale*min(ys), rescale*max(ys), ny)
    logo_dataarray = xr.DataArray(logo_values, coords=dict(x=xs_logo, y=ys_logo))

    logo_interp = logo_dataarray.interp(x=xs, y=ys)

    return np.nan_to_num(logo_interp.values, nan=np.min(logo_interp))

# %%
xs = ys = np.linspace(-design_region_width / 2, design_region_width / 2, Nx)
intensity_desired = intensity_desired_fn_logo(xs,ys)

# %%
plt.figure()
plt.pcolormesh(xs, ys, intensity_desired.T, cmap="magma")
plt.gca().set_aspect("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("desired intensity pattern")
plt.colorbar()
plt.show()

# %%
######MONITOR######
far_field_obs_point_z_coord = 10

plane_points_array = np.array([[x_p,y_p] for x_p in xs for y_p in ys])
far_z_plane = [mp.Vector3(x_p,y_p,far_field_obs_point_z_coord) for x_p in xs for y_p in ys] 
plane_points_matrix = plane_points_array.reshape(Nx,Ny,2)
#make a plot of the sampling coordinates
plt.scatter(plane_points_array[:,0], plane_points_array[:,1], s=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("sampling coordinates for near-to-far transformation")

#define the near-field observation plane (that spans the whole design region)
NearRegions = [mp.Near2FarRegion(
        center=mp.Vector3(0,0, design_region_height + wavelength),
        size=mp.Vector3(design_region_width,design_region_length,0),
        weight=+1,
)]

#mpa.Near2FarFields return a numpy array with shape (num_of_points, nfreq, 6) where the 
#third axis are the field components Ex (0),Ey (1),Ez (2),Hx (3),Hy (4),Hz (5)
FarFields = mpa.Near2FarFields(sim, NearRegions, far_z_plane)
ob_list = [FarFields]

# %%
######INTENSITIES######
def get_intensities(FF:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Utility function to compute target and measured intensities given far field data.
    Input is a numpy array with shape (num_of_points, nfreq, 6) where the 
    third axis are the field components Ex, Ey, Ez, Hx, Hy, Hz. num_of_points represents the number
    of far-field points that were passed in the list far_z_plane when defining the Near2FarRegions.
    
    Parameters:
        FF: np.ndarray with shape (num_of_points, nfreq, 6)
    '''
    #first, retrieve the target intensities from the target
    target_intensities = intensity_desired_fn_logo(xs,ys)
    #second, compute the measured intensities from the far-field
    #TODO need to check if the measured matrix needs to be flipped up/down or
    #other transformations due to how it is constructed.
    measured_intensities = npa.reshape(npa.abs(FF[:,:,0])**2, (Nx,Ny))
    
    return target_intensities, measured_intensities

# %%
######NORMALIZATION######
#compute the initial far-field intensity without 
#the presence of the metasurface to use as normalization
norm_sim = mp.Simulation(
    cell_size = cell_size,
    boundary_layers = pml_layers,
    geometry = geometry,
    sources = source,
    default_material = Air,
    resolution = resolution,
)

norm_near2far = norm_sim.add_near2far(fcen,0,1,*NearRegions)

norm_sim.plot2D(output_plane=output_plane,show_sources=True, show_boundaries=True, show_geometry=True)
plt.savefig("results/normalization_simulation_geometry.pdf")

norm_sim.run(until=sim_run_time)
ref_fields = np.array([norm_sim.get_farfield(norm_near2far, point) for point in far_z_plane])
#TODO this todo is linked to the one above. Need to check if the far-field need to be flipped or something
#at this time, the two definition are consistent with each other.
#NOTE: read carefully the docs for get_farfield.
#it returns a list of the type [Ex1,Ey1,Ez1,Hx1,Hy1,Hz1,Ex2,Ey2,Ez2,Hx2,Hy2,Hz2....]
#for frequencies 1,2...
intensity_norm_mean = npa.mean(npa.abs(ref_fields[:,0])**2) #get the Ex component for all elements in the the list
print("intensity_norm_mean shape (should be 1)",intensity_norm_mean.shape)

# %%
#this the objective to function
#the objective is to minimize the distance between the measured intenstity
# and the target.
intensity_range = int_min, int_max = (0.0, 1.0)

def J1(FF:np.ndarray):
    '''Computes the objective function for the given input field
    Parameters: 
        FF: np.ndarray with shape (num_of_points, nfreq, 6)
    Returns:
        the value of the objective function, a float
    '''
    print("FF shape", FF.shape)
    target_intensity, measured_intensity = get_intensities(FF)

    measured_intensity = measured_intensity * npa.mean(target_intensity) / intensity_norm_mean

    int_range_magnitude = npa.abs(int_max - int_min)
    measured_intensity = npa.clip(measured_intensity, int_min, int_max)
    target_intensity = int_range_magnitude * intensity_desired + int_min

    difference = measured_intensity - target_intensity
    difference_denominator = npa.full(target_intensity.shape, int_range_magnitude)

    #obj_to_return = npa.mean(npa.abs(FF[0,:,0]) ** 2)

    return npa.linalg.norm(difference) / npa.linalg.norm(difference_denominator)
#returns the first point of all frequencies of the Ex component

# %%
######OPTIMIZATION PROBLEM######
'''
An OptimizationProblem class object knows how to do one basic thing: 
Given an input vector of design variables, compute the objective function value 
(forward calculation) and optionally its gradient (adjoint calculation)
The gradient of the objective function is computed against its arguments, which are
the FarFields.
'''
opt = mpa.OptimizationProblem(
    simulation=sim, #the optimizer uses this simulation object
    objective_functions=[J1], #the optimizer call this objective function
    objective_arguments=ob_list, #the optimizer passes this list of arguments to the objective function
    design_regions=[design_region], #the optimizer updates the design variables in this design region
    frequencies=frequencies, #the optimizer runs the simulation at these frequencies
    maximum_run_time=100, #maximum optimizer run time
)
######CHECK PLOT######
figure = opt.plot2D(True, output_plane=output_plane)
plt.savefig("results/simulation_setup.pdf")

# %%
def mapping(x,eta,beta):
    '''
    Arguments:
        x: design variables (grey scale)
        eta: thresholding point for projection (between 0 and 1)
        beta: binarization strength for projection (positive)
    Returns:
        projected_field: the filtered and projected design variables
    '''
    #mapping is necessary to reach a physical design.
    #we start from the design variables (grey scale), which are values in the range (0,1).
    #typically, we set an initial value of 0.5.
    #filtering makes a local point behave more like its neighbourhood, avoiding checker patterns
    filtered_field = mpa.conic_filter(
        x,
        filter_radius,
        design_region_width,
        design_region_length,
        [design_region_resolution],
    )

    #projection
    #this is an hyperbolic tangent projection
    #after filtering, the design is still blurry, so we project it
    #using an tanh projection method to push the values towards the upper/lower bound
    #If a filtered value is >0.5, then we push it to 1, otherwise, if <0.5, we push it to 0.
    #parameter beta represents the degree of binarization,
    #parameter eta is threshold point
    projected_field = mpa.tanh_projection(filtered_field,beta,eta)
     
    #the following lines introduce symmetry. Need to understand if necessary 
    #projected_field = (
    #    npa.flipud(projected_field) + projected_field
    #)/2

    #interpolate to actual materials
    return projected_field.flatten()

evaluation_history = []
cur_iter = [0]

def f(v, gradient, cur_beta):
    print("current iteration: {}".format(cur_iter[0]+1))

    #value and grad
    f0, dJ_du = opt([mapping(v,eta_i,cur_beta)])
    if gradient.size > 0:
        # NOTE: gradient is assigned in-place! No need to return.
        gradient[...] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, cur_beta, dJ_du #NOTE: here we are using a single frequency!!
        )
    
    evaluation_history.append(np.real(f0))
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    opt.plot2D(
        False, 
        ax=ax1,
        output_plane=output_plane,
        show_sources = False,
        show_monitors = False,
        show_boundary_layers = False,
        )
    circ1 = Circle((2,2), minimum_length/2)
    ax1.add_patch(circ1)
    
    opt.plot2D(
        False, 
        ax=ax2,
        output_plane=output_plane_xy,
        show_sources = False,
        show_monitors = False,
        show_boundary_layers = False,
        )
    circ2 = Circle((2,2), minimum_length/2)
    ax2.add_patch(circ2)
    
    ax3.plot(evaluation_history)
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("objective function value")
    ax3.set_title("objective function history")

    plt.savefig("results/iter_{}.pdf".format(cur_iter[0]+1))

    cur_iter[0] += 1

    return np.real(f0)


algorithm = nlopt.LD_MMA
n = Nx * Ny #number of parameters to optimize

x = np.ones((n,)) * 0.5 #initial guess
#x = np.insert(x, 0, -1)
#lower and upper bounds for design variables
lb = np.zeros((Nx * Ny,))
ub = np.ones((Nx * Ny,))

cur_beta = 4 #starting beta
beta_scale = 2 #beta scaling (update) factor
num_betas = 3 #number of betas to test
update_factor = 1 #number of iterations per beta
#total number of simulations = num_betas * update_factor * 2 (forward + adjoint)
ftol = 1e-5

optimization_start = True
if optimization_start:
    for iters in range(num_betas):
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_min_objective(lambda v, g: f(v, g, cur_beta))
        solver.set_maxeval(update_factor) #number of iterations for each beta
        solver.set_ftol_rel(ftol)
        #observe that x is updated in place, so at
        # the end it will contain the optimized parameters
        x[...] = solver.optimize(x)
        cur_beta = cur_beta * beta_scale #update beta
        print("current beta: ", cur_beta)

# %%
######RESULTS######
#plot the figure of merit over iterations
plt.figure()
plt.plot(evaluation_history)
plt.xlabel("Iteration")
plt.ylabel("Figure of Merit")
plt.grid()
plt.savefig("results/figure_of_merit.pdf")

#plot and save the final design
np.savetxt("results/final_design.txt", mapping(x, eta_i, cur_beta))
opt.update_design([mapping(x, eta_i, cur_beta)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
opt.plot2D(
    False, 
    ax=ax1,
    output_plane=output_plane,
    show_sources = False,
    show_monitors = False,
    show_boundary_layers = False,
)

circ1 = Circle((2,2), minimum_length/2)
ax1.add_patch(circ1)
ax1.axis("off")

opt.plot2D(
    False, 
    ax=ax2,
    output_plane=output_plane_xy,
    show_sources = False,
    show_monitors = False,
    show_boundary_layers = False,
)
circ2 = Circle((2,2), minimum_length/2)
ax2.add_patch(circ2)
ax2.axis("off")
    
plt.savefig("results/final_design_xy.pdf")

# %%
######FULL SIMULATION######
#after completing the optimization we define the full domain and 
#verify that the intensity at the desired location matches the target.
#Recall that the far field plane was defined at a distance d=10
opt.update_design([mapping(x, eta_i, cur_beta)])

full_Sz = 26

opt.sim = mp.Simulation(
    cell_size = mp.Vector3(Sx,Sy,full_Sz),
    boundary_layers = pml_layers,
    k_point = mp.Vector3(),
    geometry = geometry,
    sources = source,
    default_material = Air,
    resolution = resolution,
)

src = mp.ContinuousSource(frequency=fcen, fwidth=fwidth)
source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
opt.sim.change_sources(source)

full_output_plane = mp.Volume(
    center=mp.Vector3(0,0,0), 
    size=mp.Vector3(Sx,0,full_Sz))

design_output_plane = mp.Volume(
    center=mp.Vector3(0,0,10),
    size=mp.Vector3(design_region_width,design_region_length,0)
)

#understand how to retrieve the far-field data at the design output plane
opt.sim.plot2D(output_plane=full_output_plane,fields=mp.Ex, show_sources=True, show_boundaries=True, show_geometry=True)
plt.savefig("results/full_simulation_geometry.pdf")
opt.sim.run(until=sim_run_time)