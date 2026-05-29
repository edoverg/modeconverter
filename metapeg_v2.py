
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
mp.verbosity(3)

if mp.am_master():
    if not os.path.exists("results"):
        os.makedirs("results")

###### PHYSICS SETUP ######
RESOLUTION = 20 #pixels per meep unit length (1um)
MAX_RUN_TIME = 700 #meep units
WAVELENGTH_MIN_UM = 1.50
WAVELENGTH_MAX_UM = 1.60
PML_UM = 1.0 #PML thickness
BUFFER = 3.0
DESIGN_WAVELENGTHS_UM = (1.50,1.55,1.60) #wavelengths at which to optimize the device
DESIGN_REGION_UM = mp.Vector3(4,4,1)
DESIGN_REGION_RESOLUTION = int(RESOLUTION)
DESIGN_REGION_CENTER = mp.Vector3(0, 0, DESIGN_REGION_UM.z / 2)
NX_DESIGN_GRID = int(DESIGN_REGION_UM.x * DESIGN_REGION_RESOLUTION) + 1 
NY_DESIGN_GRID = int(DESIGN_REGION_UM.y * DESIGN_REGION_RESOLUTION) + 1
MIN_LENGTH_UM = 0.5 #500 nm minimum length scale
SILICON = mp.Medium(index = 3.5)
SILICON_DIOXIDE = mp.Medium(index = 1.5)
AIR = mp.Medium(index = 1.0)

SIGMOID_THRESHOLD_INTRINSIC = 0.5
SIGMOID_THRESHOLD_EROSION = 0.75
SIGMOID_THRESHOLD_DILATION = 1 - SIGMOID_THRESHOLD_EROSION

cell_um = mp.Vector3(
    DESIGN_REGION_UM.x + 2*BUFFER + 2*PML_UM,
    DESIGN_REGION_UM.y + 2*BUFFER + 2*PML_UM,
    2*BUFFER + 2*PML_UM
)

SUBSTRATE_THICKNESS = BUFFER + PML_UM
SUBSTRATE_SIZE = mp.Vector3(cell_um.x, cell_um.y, SUBSTRATE_THICKNESS)
SUBSTRATE_CENTER = mp.Vector3(0, 0, -SUBSTRATE_THICKNESS/2)

physical_domains_size = mp.Vector3(
    DESIGN_REGION_UM.x + 2*BUFFER,
    DESIGN_REGION_UM.y + 2*BUFFER,
    2*BUFFER
)

filter_radius_um = mpa.get_conic_radius_from_eta_e(
    MIN_LENGTH_UM, SIGMOID_THRESHOLD_EROSION
)

frequency_min = 1 / WAVELENGTH_MAX_UM
frequency_max = 1 / WAVELENGTH_MIN_UM
frequency_center = 0.5 * (frequency_min + frequency_max)
wavelength_center = 1 / frequency_center #remember that c = 1 in meep units
frequency_width = frequency_max - frequency_min
frequencies = [1/wavelength for wavelength in DESIGN_WAVELENGTHS_UM]
num_wavelengths = len(DESIGN_WAVELENGTHS_UM)

SOURCE_CENTER = mp.Vector3(0,0,-BUFFER/2)
SOURCE_SIZE = mp.Vector3(physical_domains_size.x,physical_domains_size.y,0)

NEAR_REGION_MONITOR_CENTER = mp.Vector3(0,0, DESIGN_REGION_UM.z + BUFFER/2)
NEAR_REGION_MONITOR_SIZE = mp.Vector3(physical_domains_size.x,physical_domains_size.y,0)

FF_MONITOR_CENTER = mp.Vector3(0,0,DESIGN_REGION_UM.z + BUFFER)
FF_MONITOR_SIZE = mp.Vector3(DESIGN_REGION_UM.x, DESIGN_REGION_UM.y, 0)
xs = ys = np.linspace(-FF_MONITOR_SIZE.x/2, FF_MONITOR_SIZE.x/2, NX_DESIGN_GRID)
#collection of points in the far-field monitor
ff_points = [mp.Vector3(x_p,y_p,FF_MONITOR_CENTER.z) for x_p in xs for y_p in ys] 

stop_cond = mp.stop_when_fields_decayed(25, mp.Ex, NEAR_REGION_MONITOR_CENTER, 1e-6)

pml_layers = [mp.PML(PML_UM)]

view_2D_plane = mp.Volume(
    center=mp.Vector3(0,0,0), 
    size=mp.Vector3(cell_um.x,0,cell_um.z)
)

def str_from_list(list_: List[float]) -> str:
    return "[" + ", ".join(f"{val:.4f}" for val in list_) + "]"

def filter_and_project(
    weights: np.ndarray,
    sigmoid_threshold: float,
    sigmoid_bias: float
) -> np.ndarray:
    '''Function to filter and project design variables. Differentiable by autograd
    
    Args:
        weights: design weights as a flattened (1D) array
        sigmoid_threshold: erosion/dilation parameter for projection
        sigmoid_bias: bias parameter for the projection. 0 is no projection
    Returns:
        The flattened (1D) filtered/projected design weights
    '''
    
    print("Inside filter_and_project function...")
    weights_filtered = mpa.conic_filter(
        weights.reshape(-1, NY_DESIGN_GRID), #row major mapping 
        filter_radius_um,
        DESIGN_REGION_UM.x,
        DESIGN_REGION_UM.y,
        DESIGN_REGION_RESOLUTION,
    )

    if sigmoid_bias == 0:
        print("No projection applied in filter_and_project function, exiting...")
        return weights_filtered.flatten()
    else:
        weights_projected = mpa.tanh_projection(
            weights_filtered.reshape(-1,NY_DESIGN_GRID),
            sigmoid_bias,
            sigmoid_threshold,
        )
        print("Projection applied in filter_and_project function, exiting...")
        return weights_projected.flatten()
    
def obj_fun(epigraph_and_weights: np.ndarray, grad: np.ndarray)-> float:
    '''Objective function for the epigraph formulation
    
    Args:
        epigraph_and_weights: 1D array containing epigraph variable
        (at index 0) and design weigts (remaining elements)
        grad: the gradient modified in place of the objective function (flattened 1D array)
    Returns:
        The scalar epigraph variable (the objective function)
    '''

    epigraph = epigraph_and_weights[0]
    
    if grad.size > 0:
        grad[0] = 1
        grad[1:] = 0

    return epigraph

def epigraph_constraint(
        result: np.ndarray,
        epigraph_and_weights: np.ndarray,
        gradient: np.ndarray,
        sigmoid_threshold: float,
        sigmoid_bias: float,
        use_epsavg: bool,
) -> None:
    '''Constraint function for the epigraph formulation

    Args:
        result: 1D array, modified in place, with the results 
            of this constraint evaluation
        epigraph_and_weights: 1D array of epigraph (first element) 
            and design weights (other elements)
        gradient: the jacobian matrix (backpropagated) with dimensions
            (1+NX_DESIGN_GRID*NY_DESIGN_GRID, 2 * num_wavelengths), modified in place.
            these gradients are used by the optimization algorithm,
            because, being backpropagated, they tell us how the constraint function behaves
            with respect to changes in the design parameters.
        sigmoid_threshold: erosion/dilation parameter for projection.
        sigmoid_bias: bias parameter for projection.
        use_epsavg: whether to use subpixel smoothing.
    '''
    print("Inside epigraph_constraint function...")
    epigraph = epigraph_and_weights[0]
    weights = epigraph_and_weights[1:]
    assert not np.isnan(weights).any(), "NaN values found in weights in epigraph_constraint function"
    print("===========")
    print("Shape of weights in epigraph_constraint:", weights.shape)
    print("===========")
    obj_val, grad = opt(
        [
            filter_and_project(
                weights, sigmoid_threshold, 0 if use_epsavg else sigmoid_bias
            )
        ]
    )
    print("filter_and_project called")
    #in our case there is only one objective function, 
    #which needs to be evaluated at multiple wavelengths.
    #for this reason, obj_val is a 1-element list (to be verified,
    # it might be that the list is directly replaced by the content)
    # that contains an array of objective function values at each wavelength.
    print("Backpropagating gradients...")
    #for each wavelength, backpropagate
    for k in range(num_wavelengths):
        grad[:,k] = tensor_jacobian_product(filter_and_project, 0)(
            weights,
            sigmoid_threshold,
            sigmoid_bias,
            grad[:,k],
        )

    #modify in place the gradient result
    if gradient.size > 0:
        gradient[:,0] = -1 #gradient w.r.t epigraph
        gradient[:,1:] = grad.T

    #modify in place the constraint result
    result[:] = np.real(obj_val) - epigraph

    print("Updating history...")
    objfunc_history.append(np.real(obj_val))
    epivar_history.append(epigraph)

    print(
        f"iteration:, {cur_iter[0]:3d}, sigmoid_bias: {sigmoid_bias:2d}, "
        f"epigraph: {epigraph:.5f}, obj. func.: {obj_val}, "
        f"epigraph constraint: {str_from_list(result)}"
    )
    print("Finished epigraph_constraint function")
    cur_iter[0] = cur_iter[0] + 1

def line_width_and_spacing_constraint(
        result: np.ndarray,
        epigraph_and_weights: np.ndarray,
        gradient: np.ndarray,
        sigmoid_bias: float,
) -> float:
    '''Constraint function for minimum line width and spacing

    Args: 
        result: evaluation of this constraint function, modified in place
        epigraph_and_weights: 1D array of epigraph (first element) 
            and design weights (other elements)
        gradient: the Jacobian matrix, modified in place
        sigmoid_bias: bias parameter for projection
    
    Returns:
        The constraint value result
    '''

    epigraph = epigraph_and_weights[0]
    weights = epigraph_and_weights[1:]

    a1 = 1e-3
    b1 = 0
    gradient[:,0] = -a1

    filter_func = lambda a: mpa.conic_filter(
            a.reshape(-1,NY_DESIGN_GRID),
            filter_radius_um,
            DESIGN_REGION_UM.x,
            DESIGN_REGION_UM.y,
            DESIGN_REGION_RESOLUTION,
        )
    
    threshold_func = lambda a: mpa.tanh_projection(
            a.reshape(-1,NY_DESIGN_GRID),
            sigmoid_bias,
            SIGMOID_THRESHOLD_INTRINSIC,
    )

    c0 = 1e7 * (filter_radius_um * 1 / RESOLUTION) ** 4

    #constraints as lambda functions
    M1 = lambda a: mpa.constraint_solid(
        a,
        c0,
        SIGMOID_THRESHOLD_EROSION,
        filter_func,
        threshold_func,
        1
    )

    M2 = lambda a: mpa.constraint_void(
        a,
        c0,
        SIGMOID_THRESHOLD_DILATION,
        filter_func,
        threshold_func,
        1
    )

    #gradients of the constraints w.r.t design parameters
    g1 = grad(M1)(weights)
    g2 = grad(M2)(weights)

    result[0] = M1(weights) - a1 * epigraph - b1
    result[1] = M2(weights) - a1 * epigraph - b1

    gradient[0, 1:] = g1.flatten()
    gradient[1, 1:] = g2.flatten()

    #these are the epigraph values for the constraints
    t1 = (M1(weights) - b1) / a1
    t2 = (M2(weights) - b1) / a1

    print(f"line_width_and_spacing_constraint:, {result[0]}, {result[1]}, {t1}, {t2}")

    return max(t1, t2)

def get_pattern() -> np.ndarray:
    '''Imports the intensity pattern from file'''
    pattern_name = 'peg.png'
    pattern = Image.open('misc/' + pattern_name).convert('L')
    pattern = np.array(pattern).astype(float)

    pattern = pattern - np.min(pattern)
    pattern = pattern / np.max(pattern)

    return pattern

def intensity_desired_fn_pattern(
    xs: list,
    ys: list,
    rescale: float = 0.5
) -> np.ndarray:
    '''Returns the pattern intensity value at the specified
    (xs,ys) locations, with a scale factor
    
    Args: 
        xs: list of x coordinates
        ys: list of y coordinates
        rescale: scale factor for the pattern
    Returns:
        The flattened pattern intensity value at the specified (xs,ys) locations, with a scale factor
    '''
    pattern_values = get_pattern()
    pattern_values = 1 - np.rot90(np.rot90(np.rot90(pattern_values)))

    nx,ny = pattern_values.shape
    xs_pattern = np.linspace(rescale*min(xs), rescale*max(xs), nx)
    ys_pattern = np.linspace(rescale*min(ys), rescale*max(ys), ny)
    pattern_dataArray = xr.DataArray(pattern_values, coords=dict(x=xs_pattern, y=ys_pattern))
    pattern_interp = pattern_dataArray.interp(x=xs, y=ys)

    pattern_final = npa.nan_to_num(pattern_interp.values, nan=npa.min(pattern_interp))
    pattern_to_return = npa.nan_to_num(pattern_interp.values, nan=npa.min(pattern_interp)).flatten()
    
    #make a plot before returning
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(
        pattern_final,
        extent=(min(xs), max(xs), min(ys), max(ys)),
        origin='lower',
        cmap='inferno',
    )
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_title('Target Intensity Pattern')
    plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
    plt.savefig("results/target_intensity_pattern.pdf")

    return pattern_to_return

def normalization_sim() -> np.ndarray:
    ''' Computes the Far-Field pattern at the monitor location and
    without the design region. Used for normalization purposes. 
    
    Returns:
        A 2D-array [Nx*Ny,num_wavelengths] of reference intensities, for each wavelength,
          at the monitor location.
    '''

    sources = [
        mp.Source(
            src=mp.GaussianSource(
                frequency=frequency_center, 
                fwidth=frequency_width),
            component=mp.Ex,
            center=SOURCE_CENTER,
            size=SOURCE_SIZE,
        )
    ]

    substrate_block = mp.Block(
        center = SUBSTRATE_CENTER,
        size = SUBSTRATE_SIZE,
        material = SILICON_DIOXIDE,
    )

    geometry = [#combine geometries
        substrate_block,
    ]

    norm_sim = mp.Simulation(
        resolution=RESOLUTION,
        default_material=AIR,
        cell_size=cell_um,
        sources=sources,
        geometry=geometry,
        boundary_layers=pml_layers,
        k_point=mp.Vector3(),
    )

    #plt.figure()
    #ax = plt.gca()
    #norm_sim.plot2D(output_plane=view_2D_plane, ax=ax)
    #plt.savefig("results/normalization_sim_setup.pdf")
    #plt.close()

    NearRegions = [mp.Near2FarRegion(
        center=NEAR_REGION_MONITOR_CENTER,
        size=NEAR_REGION_MONITOR_SIZE,
        weight=+1,
    )]

    norm_near2far = norm_sim.add_near2far(frequencies, *NearRegions)

    norm_sim.run(
        #mp.at_every(1,mp.in_volume(
        #    mp.Volume(center=mp.Vector3(), size=physical_domains_size),
        #    mp.output_efield_x
        #)),
        #mp.at_every(20,record_fields),
        until_after_sources=stop_cond
    )
    #make a plot of the fields when simulation finished
    #plt.figure()
    #ax = plt.gca()
    #norm_sim.plot2D(output_plane=view_2D_plane, fields=mp.Ex, ax=ax)
    #plt.savefig("results/fields_after_norm_sim.pdf")
    #plt.close()

    ref_fields = np.array(
        [norm_sim.get_farfield(norm_near2far, point) for point in ff_points]
    )
    
    #select only the Ex component for all frequencies, 
    # which is the one we will optimize for
    ref_fields_x_f1 = ref_fields[:,0]
    ref_fields_x_f2 = ref_fields[:,6]
    ref_fields_x_f3 = ref_fields[:,12]
    #concatenate the Ex component for all frequencies into a single array
    #the rows are the points, the columns are the frequencies
    ref_fields_x = npa.stack((ref_fields_x_f1, ref_fields_x_f2, ref_fields_x_f3), axis=-1)
    
    ref_intensity_x = npa.power(npa.abs(ref_fields_x),2)
    return ref_intensity_x
    
def intensity_from_farfields(FarFields) -> np.ndarray:
    '''Computes the intensity pattern at the monitor location from the simulated far-fields

    Args:
        FarFields: the simulated far-field patterns at the farfield 
        monitor location, for each wavelength
    Returns:
        A 2D-array [Nx*Ny,num_wavelengths] of intensities, for each wavelength,
          at the monitor location.
    '''
    #select only the Ex component for all frequencies, 
    # which is the one we will optimize for.
    #Reshape into matrix form (Nx,Ny)
    #print(FarFields.shape)
    #fields_x_f1 = FarFields[:,0,0].reshape(NX_DESIGN_GRID, NY_DESIGN_GRID)
    #fields_x_f2 = FarFields[:,1,0].reshape(NX_DESIGN_GRID, NY_DESIGN_GRID)
    #fields_x_f3 = FarFields[:,2,0].reshape(NX_DESIGN_GRID, NY_DESIGN_GRID)
    #concatenate the Ex component for all frequencies into a single array
    #the rows are the points, the columns are the frequencies
    #make a 3d arrray with dimensions (Nx,Ny,num_wavelengths)
    #fields_stack_3d = npa.stack((fields_x_f1, fields_x_f2, fields_x_f3), axis=-1)
    #fields_x = np.stack((fields_x_f1, fields_x_f2, fields_x_f3), axis=-1)
    
    intensity_x = npa.abs(FarFields[:,:,0]) ** 2

    return intensity_x

def intensity_optimization(
        ref_intensity: np.ndarray,
        use_damping:bool,
        use_epsavg: bool,
        sigmoid_bias: float,
)-> mpa.OptimizationProblem:
    '''Sets up the optimization problem for the intensity pattern matching

    Args:
        ref_intensity: 2D array of reference intensities at the monitor location, for each wavelength
        use_damping: whether to use damping in the optimization algorithm
        use_epsavg: whether to use subpixel smoothing in the optimization
        sigmoid_bias: bias parameter for projection
    Returns:
        The optimization problem class instance
    '''
    matgrid = mp.MaterialGrid(
        grid_size = mp.Vector3(NX_DESIGN_GRID, NY_DESIGN_GRID,0),
        medium1 = AIR,
        medium2 = SILICON,
        weights = np.ones((NX_DESIGN_GRID, NY_DESIGN_GRID)),
        beta = sigmoid_bias if use_epsavg else 0,
        do_averaging = True if use_epsavg else False,
        damping = 0.02*2*np.pi*frequency_center if use_damping else 0,
    )

    matgrid_region = mpa.DesignRegion(
        matgrid,
        volume=mp.Volume(center=DESIGN_REGION_CENTER, 
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

    #2D array of target intensity at the monitor location
    target_intensity = intensity_desired_fn_pattern(xs, ys)
    target_intensity_3d = npa.tile(target_intensity[:,npa.newaxis], (1,num_wavelengths))
    target_intensity_mean = npa.mean(target_intensity)

    def J1(FarFields):
        '''Objective function. Computes the distance between simulated and target intensity patterns.
        Args:
            FarFields: the simulated far-field patterns at the monitor location, for each wavelength
        Returns:
            The distance between simulated and target intensity patterns
        '''
 
        measured_intensity = intensity_from_farfields(FarFields)
        
        #measured_intensity_mean = npa.mean(measured_intensity, axis=0)

        translated_measured_intensity = measured_intensity / npa.mean(ref_intensity,axis=0) * target_intensity_mean
        
        translated_measured_intensity = npa.clip(translated_measured_intensity, a_min=0, a_max=1)

        distance = translated_measured_intensity - target_intensity_3d
        
        return npa.power(npa.sum(distance**2, axis=0),0.5)
        

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=[J1],
        objective_arguments=obj_list,
        design_regions=[matgrid_region],
        frequencies=frequencies,
        maximum_run_time=MAX_RUN_TIME
    )

    plt.figure()
    ax = plt.gca()
    opt.plot2D(True,output_plane=view_2D_plane, ax=ax)
    plt.savefig("results/optimization_problem_setup.pdf")
    
    return opt

if __name__ == "__main__":
    print("Hello there! Starting normalization")
    ref_intensity = normalization_sim()
    
    num_weights = NX_DESIGN_GRID * NY_DESIGN_GRID

    epigraph_and_weights = np.ones((num_weights,)) * 0.5
    epigraph_and_weights = np.insert(epigraph_and_weights, 0, 1.2) 

    weights_lower_bound = np.zeros(num_weights)
    weights_upper_bound = np.ones(num_weights)
    epigraph_and_weights_lower_bound = np.insert(weights_lower_bound, 0, -np.inf)
    epigraph_and_weights_upper_bound = np.insert(weights_upper_bound, 0, +np.inf)

    objfunc_history = []
    epivar_history = []
    cur_iter = [0]

    sigmoid_bias_threshold = 64

    #sigmoid_biases = [8, 16, 32, 64, 128, 256]
    sigmoid_biases = [8, 16, 32]
    #max_evals = [48, 48, 60, 60, 60, 60]
    max_evals = [48, 48, 60]
    
    epigraph_tolerance = np.array([1e-4]*num_wavelengths)
    tolerance_width_and_spacing = np.array([1e-8]*2)

    for sigmoid_bias, max_eval in zip(sigmoid_biases, max_evals):
        print(f"Starting optimization epoch with sigmoid bias {sigmoid_bias} and max evals {max_eval}")
        #the optimizer needs to work with the weights and epigraph (+1)
        solver = nlopt.opt(nlopt.LD_CCSAQ, num_weights + 1)
        solver.set_lower_bounds(epigraph_and_weights_lower_bound)
        solver.set_upper_bounds(epigraph_and_weights_upper_bound)
        solver.set_min_objective(obj_fun)
        solver.set_maxeval(max_eval)
        solver.set_param("dual_ftol_rel", 1e-7)
        solver.add_inequality_mconstraint(
            lambda result_, epigraph_and_weights_, grad_: epigraph_constraint(
                result_, 
                epigraph_and_weights_, 
                grad_, 
                SIGMOID_THRESHOLD_INTRINSIC, 
                sigmoid_bias, 
                use_epsavg=False if sigmoid_bias < sigmoid_bias_threshold else True,
            ),
            epigraph_tolerance,
        )
        solver.set_param("verbosity",1)
        if sigmoid_bias < sigmoid_bias_threshold:
            use_epsavg = False
        else:
            use_epsavg = True
        
        opt = intensity_optimization(
            ref_intensity,
            use_damping=True,
            use_epsavg=use_epsavg,
            sigmoid_bias=sigmoid_bias,
        )

        #the minimum linewidth and spacing constraint
        #is activated only in the last epoch. This is done
        #by adding the corresponding constraints to the same solver object.
        #constraints are added as lambda functions, which
        #are called directly by the nlopt solver.
        if sigmoid_bias == sigmoid_biases[-1]: #if we are at the last sigmoid bias...
            line_width_and_spacing = np.zeros(2)
            grad_line_width_and_spacing = np.zeros((2, num_weights + 1))
            #compute the value of the costraints at this time
            linewidth_constraint_val = line_width_and_spacing_constraint(
                line_width_and_spacing,
                epigraph_and_weights,
                grad_line_width_and_spacing,
                sigmoid_bias,
            )
            solver.add_inequality_mconstraint(
                lambda result_, epigraph_and_weights_, grad_: line_width_and_spacing_constraint(
                    result_,
                    epigraph_and_weights_,
                    grad_,
                    sigmoid_bias,
                ),
                tolerance_width_and_spacing,
            )

        #execute a SINGLE forward run before the start of each epoch and
        #manually set the initial epigraph variable to slightly larger
        #than the largest value of the objective function over the 
        #wavelengths and the lengthscale constraint
        #making a call to opt like the following invokes 
        #the __call__ method of the opt instance and starts the optimization
        print("Calibrating epigraph variable before starting optimization...")
        epigraph_initial, empty_grads = opt(
            [
                filter_and_project(
                    epigraph_and_weights[1:], 
                    SIGMOID_THRESHOLD_INTRINSIC,
                    sigmoid_bias if sigmoid_bias < sigmoid_bias_threshold else 0,
                ),
            ],
            need_gradient=False,
        )
        #epigraph_initial contains the objective function values 
        #at each wavelength in the simulation. At each epoch,
        #we set the initial epigraph variable taking the max of the objective functions
        epigraph_and_weights[0] = np.max(epigraph_initial)
        fraction_max_epigraph = 0.05
        if sigmoid_bias == sigmoid_biases[-1]:#if last epoch
            epigraph_and_weights[0] = \
            (1 + fraction_max_epigraph) * max(
                epigraph_and_weights[0], linewidth_constraint_val)
        print(
            f"epigraph-calibration:, bias = {sigmoid_bias}, "
            f"{str_from_list(epigraph_initial)}, {epigraph_and_weights[0]}"
        )

        print("Starting optimization...")
        epigraph_and_weights[:] = solver.optimize(epigraph_and_weights)
        print("Optimization completed")

        optimal_design_weights = filter_and_project(
            epigraph_and_weights[1:],
            SIGMOID_THRESHOLD_INTRINSIC,
            sigmoid_bias
        ).reshape(-1, NY_DESIGN_GRID)

        fig, ax = plt.subplots()
        ax.imshow(
            optimal_design_weights,
            cmap="binary",
            interpolation="none",
        )
        ax.set_axis_off()
        if mp.am_master():
            fig.savefig(
                f"optimal_design_beta{sigmoid_bias}.png",
                dpi=150,
                bbox_inches="tight",
            )
            # Save the final (unmapped) design as a 2D array in CSV format
            np.savetxt(
                f"unmapped_design_weights_beta{sigmoid_bias}.csv",
                epigraph_and_weights[1:].reshape(NX_DESIGN_GRID, NY_DESIGN_GRID),
                fmt="%4.2f",
                delimiter=",",
            )

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(objfunc_history[:,0], label="f1 - 1.50 um")
    plt.plot(objfunc_history[:,1], label="f2 - 1.55 um")
    plt.plot(objfunc_history[:,2], label="f3 - 1.60 um")
    plt.subplot(1,2,2)
    plt.plot(epivar_history, label="Epigraph")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("results/optimization_history.pdf")

    saveResults = True
    if saveResults:
        with open('results/optimal_design.pkl', 'wb') as f:
            pickle.dump({
                'RESOLUTION': RESOLUTION,
                'MAX_RUN_TIME': MAX_RUN_TIME,
                'WAVELENGTH_MIN_UM': WAVELENGTH_MIN_UM,
                'WAVELENGTH_MAX_UM': WAVELENGTH_MAX_UM,
                'PML_UM': PML_UM,
                'BUFFER': BUFFER,
                'DESIGN_WAVELENGTHS_UM': DESIGN_WAVELENGTHS_UM,
                'DESIGN_REGION_UM': DESIGN_REGION_UM,
                'DESIGN_REGION_RESOLUTION': DESIGN_REGION_RESOLUTION,
                'DESIGN_REGION_CENTER': DESIGN_REGION_CENTER,
                'NX_DESIGN_GRID': NX_DESIGN_GRID,
                'NY_DESIGN_GRID': NY_DESIGN_GRID,
                'MIN_LENGTH_UM': MIN_LENGTH_UM,
                'SILICON': SILICON,
                'SILICON_DIOXIDE': SILICON_DIOXIDE,
                'AIR': AIR,

                'SIGMOID_THRESHOLD_INTRINSIC': SIGMOID_THRESHOLD_INTRINSIC,
                'SIGMOID_THRESHOLD_EROSION': SIGMOID_THRESHOLD_EROSION,
                'SIGMOID_THRESHOLD_DILATION': SIGMOID_THRESHOLD_DILATION,
                
                'cell_um': cell_um,

                'SUBSTRATE_THICKNESS': SUBSTRATE_THICKNESS,
                'SUBSTRATE_SIZE': SUBSTRATE_SIZE,
                'SUBSTRATE_CENTER': SUBSTRATE_CENTER,
                
                'physical_domains_size': physical_domains_size,
                'filter_radius_um': filter_radius_um,

                'frequency_min': frequency_min,
                'frequency_max': frequency_max,
                'frequency_center': frequency_center,
                'wavelength_center': wavelength_center,
                'frequency_width': frequency_width,
                'frequencies': frequencies,
                'num_wavelengths': num_wavelengths,

                'SOURCE_CENTER': SOURCE_CENTER,
                'SOURCE_SIZE': SOURCE_SIZE,
                
                'NEAR_REGION_MONITOR_CENTER': NEAR_REGION_MONITOR_CENTER,
                'NEAR_REGION_MONITOR_SIZE': NEAR_REGION_MONITOR_SIZE,
                
                'FF_MONITOR_CENTER': FF_MONITOR_CENTER,
                'FF_MONITOR_SIZE': FF_MONITOR_SIZE,
                'xs': xs,
                'ys': ys,
                'ff_points': ff_points,
                
                'sigmoid_biases': sigmoid_biases,
                'max_eval': max_eval,
                'objfunc_history': objfunc_history,
                'epivar_history': epivar_history,
                'epigraph_variable': epigraph_and_weights[0],
                'unmapped_design_weights': epigraph_and_weights[1:],
                'optimal_design_weights': optimal_design_weights,
            }, f)

    print("Simulation completed")
