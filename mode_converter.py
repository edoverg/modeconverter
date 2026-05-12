# This is a first attempt at creating an inverse designed fiber mode converter in meep. At first, we try to implement and combine tutorials from both tidy3d and meep.


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


######PHYSICS SETUP######

mp.verbosity(1)

#setup a results folder
if not os.path.exists("results"):
    os.makedirs("results")

subs_index = 3.4
subs_perm = subs_index**2

Si = mp.Medium(index = subs_index)
Air = mp.Medium(index = 1.0)

wavelength = 1.55
nf = 3 #number of frequencies to test
frequencies = np.array([1/1.5, 1 / 1.55, 1 / 1.6])

#distance between PML and source/monitor
buffer = 2.5

######PML######
pml_size = 1.0
pml_layers = [mp.PML(pml_size)]


######FILTER######
minimum_length = 0.09  # minimum length scale (microns)
eta_i = (
    0.5  # blueprint (or intermediate) design field thresholding point (between 0 and 1)
)
eta_e = 0.55  # erosion design field thresholding point (between 0 and 1)
eta_d = 1 - eta_e  # dilation design field thresholding point (between 0 and 1)
filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)


######DESIGN REGION######
design_region_width = 4.0
design_region_length = 4.0
design_region_height = 1.0

resolution = 15 # pixels/um
design_region_resolution = int(resolution)

#number of pixels in the design region
Nx = int(design_region_width * design_region_resolution) + 1
Ny = int(design_region_length * design_region_resolution) + 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type="U_MEAN")

design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(
        center = mp.Vector3(0,0,design_region_height / 2),
        size = mp.Vector3(design_region_width, 
                          design_region_length, 
                          design_region_height),
    ),
)

######CELL GEOMETRY######
Sx = 2 * pml_size + design_region_width
Sy = 2 * pml_size + design_region_length
Sz = 2 * pml_size + design_region_height + 2*buffer
print("Cell size: ", Sx, Sy, Sz)
cell_size = mp.Vector3(Sx, Sy, Sz)

substrate_thickness = Sz/2
substrate_size = mp.Vector3(design_region_width, design_region_length, substrate_thickness)
substrate_center = mp.Vector3(
                    0,
                    0,
                    - substrate_thickness / 2)

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

geometry = [
    substrate_block,
    design_region_block,
]


######SOURCE######
fcen = 1 / wavelength
width = 0.2
fwidth = width * fcen
source_center = [0, 0, -3]
source_size = mp.Vector3(design_region_width, design_region_length, 0)
src = mp.GaussianSource(frequency=fcen, fwidth=fwidth)
source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]
#src = mp.ContinuousSource(frequency=fcen, fwidth=fwidth, is_integrated=True)
#source = [mp.Source(src, component=mp.Ex, size=source_size, center=source_center)]


######SIMULATION######
kpoint = mp.Vector3()
sim = mp.Simulation(
    cell_size = cell_size,
    boundary_layers = pml_layers,
    geometry = geometry,
    sources = source,
    default_material = Air,
    resolution = resolution,
)


######MONITOR######
far_field_obs_point_z_coord = 15
far_z = [mp.Vector3(0,0,far_field_obs_point_z_coord)] #this is the far-field-focal-point
NearRegions = [
    mp.Near2FarRegion(
        center=mp.Vector3(0,0, design_region_height + wavelength),
        size=mp.Vector3(design_region_width,design_region_length,0),
        weight=+1,
    )
]
#mpa.Near2FarFields return a numpy array with shape (num_of_points,nfreq,6) where the 
#third axis are the field components Ex,Ey,Ez,Hx,Hy,Hz
FarFields = mpa.Near2FarFields(sim,NearRegions,far_z)
ob_list = [FarFields]

#this the objective to function
#the objective is to maximize the obj function,
def J1(FF):
    #in this case we want to maximize field intensity
    # at the position of the focal spot 
    obj_to_return = npa.mean(npa.abs(FF[0,:,1]) ** 2)
    return obj_to_return
#returns the first point of all frequencies of the Ey component


######CHECK PLOT######
output_plane = mp.Volume(
    center=mp.Vector3(0,0,0), 
    size=mp.Vector3(Sx-2*pml_size,0,Sz-2*pml_size))

output_plane_xy = mp.Volume(
    center=mp.Vector3(0,0,design_region_height / 2),
    size=mp.Vector3(Sx-2*pml_size,Sy-2*pml_size,0)
)

fig = sim.plot2D(show_sources=True,show_monitors=True,show_epsilon=True, output_plane=output_plane)
#sim.run(until=10)
#sim.plot2D(fields=mp.Ex,output_plane=output_plane)
#plt.savefig("fields.pdf")


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
    maximum_run_time=500, #maximum optimizer run time
)

######CHECK PLOT######
figure = opt.plot2D(True, output_plane=output_plane)
plt.savefig("results/simulation_setup.pdf")



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
            v, eta_i, cur_beta, np.sum(dJ_du, axis=1)
        )
    
    evaluation_history.append(np.real(f0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    opt.plot2D(
        False, 
        ax=ax1,
        output_plane=output_plane,
        show_sources = True,
        show_monitors = True,
        show_boundary_layers = True,
        )
    circ1 = Circle((2,2), minimum_length/2)
    ax1.add_patch(circ1)
    
    opt.plot2D(
        False, 
        ax=ax2,
        output_plane=output_plane_xy,
        show_sources = True,
        show_monitors = True,
        show_boundary_layers = True,
        )
    circ2 = Circle((2,2), minimum_length/2)
    ax2.add_patch(circ2)
    
    plt.savefig("results/iter_{}.pdf".format(cur_iter[0]+1))

    cur_iter[0] += cur_iter[0] + 1

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
num_betas = 6 #number of betas to test
update_factor = 12 #number of iterations per beta
#total number of simulations = num_betas * update_factor * 2 (forward + adjoint)
ftol = 1e-5

optimization_start = True
if optimization_start:
    for iters in range(num_betas):
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_max_objective(lambda v, g: f(v, g, cur_beta))
        solver.set_maxeval(update_factor) #number of iterations for each beta
        solver.set_ftol_rel(ftol)
        #observe that x is updated in place, so at
        # the end it will contain the optimized parameters
        x[...] = solver.optimize(x)
        cur_beta = cur_beta * beta_scale #update beta
        print("current beta: ", cur_beta)

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
    show_sources = True,
    show_monitors = True,
    show_boundary_layers = True,
)

circ1 = Circle((2,2), minimum_length/2)
ax1.add_patch(circ1)
ax1.axis("off")

opt.plot2D(
    False, 
    ax=ax2,
    output_plane=output_plane_xy,
    show_sources = True,
    show_monitors = True,
    show_boundary_layers = True,
)
circ2 = Circle((2,2), minimum_length/2)
ax2.add_patch(circ2)
ax2.axis("off")
    
plt.savefig("results/final_design.pdf")



