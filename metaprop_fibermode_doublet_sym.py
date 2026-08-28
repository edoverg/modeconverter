import os

save_folder = "results_sym"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

lock_file_path = f"{save_folder}/lock_file.lock"
if os.path.exists(lock_file_path):
    with open(lock_file_path, "r") as lock_file:
        lock_status = lock_file.read().strip()
        if lock_status != "open":
            print("A lock is preventing this script to run.")
            exit(0)
        elif lock_status == "open":
            print("Lock is open. Continuing execution.")
else:
    print("A lock file must exist before running.")
    exit(0)

import time
import multiprocessing

import pyfftw
#pyFFTW setup
#n_cpu = multiprocessing.cpu_count()
n_cpu = 1
pyfftw.config.NUM_THREADS = n_cpu
pyfftw.interfaces.cache.disable()
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'

import nlopt
nlopt.srand(42)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#set the global font size for all plots
plt.rcParams.update({'font.size': 14})

from scipy.special import jv, kv

##############################
#physics and simulation domain
wavelength = 1.55e-6
n0 = 1.45
n1 = 1.0
n2 = 1.45

lda0 = wavelength / n0
lda1 = wavelength / n1
lda2 = wavelength / n2

k0 = 2 * np.pi / lda0
k1 = 2 * np.pi / lda1
k2 = 2 * np.pi / lda2

opt_max_eval = 150

unit_cell_pitch = 700e-9 #equivalent to spatial sampling

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

#important checks on spatial grid
try:
    assert(len(xs) == Nx)
    assert(len(ys) == Ny)
    assert(xs[0] == xs[-1]*-1)
    assert(ys[0] == ys[-1]*-1)
    assert(xs[nx]==0)
    assert(ys[ny]==0)
    assert(np.abs(np.abs(xs[1]-xs[0]) - unit_cell_pitch) < 1e-12)
except:
    raise ValueError("Error while setting up the spatial grid.")

X, Y = np.meshgrid(xs, ys)
rho = np.sqrt(X**2 + Y**2)

sampling_period = xs[1] - xs[0]

d0 = 476e-6 #propagation distance: source - MS1 [m]
d_source = d0 - lda0 / 2  #propagation distance: source - MS1 used for FDTD simulations [m]
d1 = 510e-6 #propagation distance: MS1 - MS2 [m]
d2 = d0 #propagation distance: MS2 - target [m]

phase_min = 0
phase_max = 2 * np.pi
##############################

##############################
#Source
beam_waist = 5.2e-6 #[um]
#output fibre
output_fiber_core_diameter = 19e-6 #[um]
#fibre cladding diameter
fibre_cladding_diameter = 125e-6 #[um]
max_available_diameter = 122e-6 #[um]
##############################

##############################
#plan pyfftw objects for forward and inverse FFTs
fft_input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
fft_output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
ifft_input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
ifft_output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
fft_input_array.fill(0)
fft_output_array.fill(0)
ifft_input_array.fill(0)
ifft_output_array.fill(0)
fft_operator = pyfftw.FFTW(fft_input_array, fft_output_array, axes=(0,1), direction='FFTW_FORWARD', threads=n_cpu, flags=('FFTW_ESTIMATE',))
ifft_operator = pyfftw.FFTW(ifft_input_array, ifft_output_array, axes=(0,1), direction='FFTW_BACKWARD', threads=n_cpu, flags=('FFTW_ESTIMATE',))
##############################

##############################
#setting up spatial frequencies
ks = 2 * np.pi / sampling_period
kappas = np.arange(-Nx//2+1, Nx//2+1) * ks / Nx
KX, KY = np.meshgrid(kappas, kappas)
K_parallel = np.sqrt(KX**2 + KY**2)
nu_parallel = K_parallel / (2 * np.pi)

phase_factor_0 = k0 * d0 * np.sqrt(1 - (lda0 * nu_parallel) ** 2 + 0*1j)
P_0 = np.exp(1j * phase_factor_0)
P_0_nat = np.fft.ifftshift(P_0)

phase_factor_0_source = k0 * d_source * np.sqrt(1 - (lda0 * nu_parallel) ** 2 + 0*1j)
P_0_source = np.exp(1j * phase_factor_0_source)
P_0_nat_source = np.fft.ifftshift(P_0_source)

phase_factor_1 = k1 * d1 * np.sqrt(1 - (lda1 * nu_parallel) ** 2 + 0*1j)
P_1 = np.exp(1j * phase_factor_1)
P_1_nat = np.fft.ifftshift(P_1)
phase_factor_2 = k2 * d2 * np.sqrt(1 - (lda2 * nu_parallel) ** 2 + 0*1j)
P_2 = np.exp(1j * phase_factor_2)
P_2_nat = np.fft.ifftshift(P_2)

P_1_dagger = P_1.T.conj()
P_2_dagger = P_2.T.conj()
P_1_dagger_nat = np.fft.ifftshift(P_1).T.conj()
P_2_dagger_nat = np.fft.ifftshift(P_2).T.conj()
##############################

##############################
#Plot zoom parameters [um]
zoom_x = 150
zoom_y = 150
zoom_output_fiber_radius = 63
full_view_x = size_x * 1e6 / 2
full_view_y = size_y * 1e6 / 2
##############################

def phase_given_w(w):
    '''Returns the phase shift given the design parameters w.
    Args:
        w: design parameters (flattened)
    Returns:
        The phase shift corresponding to the geometrical parameters w (flattened)'''
    #for simplicity, we linearly map w (0,1) to a phase shift (0, 2pi)
    return 2 * np.pi * (w)

def get_pattern() -> np.ndarray:
    '''Returns the flattened target fiber mode pattern'''
    pattern_name = 'LP11_field_distribution.txt'
    pattern = np.loadtxt('misc/' + pattern_name, dtype=complex)

    return pattern.flatten()

def get_fiber_mode_pattern(mode_list:list)->list:
    '''Returns the fiber mode pattern for the selected n index.
    Args:
        mode_list: the list of the selected fiber mode indices
    Returns:
        The fiber mode patterns corresponding to the selected indices returned 
        as a list of 2D arrays with the same order as the input mode_list
    '''
    #check if the modes in mode_list have already been compute by checking
    #the saved files in the 'results' folder. If they have, load them from the files. If not, compute them and save them to the files.
    fiber_modes_input_folder = "fiber_modes"
    output_patterns = []
    try:
        for enum_index, n in enumerate(mode_list):
            #try to load mode index n
            try:
                pattern_file = fiber_modes_input_folder + f"/LP{n}1_field_distribution_{Nx}.npy"
                pattern = np.load(pattern_file)
                output_patterns.append(pattern)
                print("Loaded fiber patterns for mode index: ", n)
            #if mode index n was not found, compute it
            except FileNotFoundError:
                if not os.path.exists(fiber_modes_input_folder):
                        os.makedirs(fiber_modes_input_folder)

                print("Computing fiber patterns for mode index: ", n)
                lda = 1.55e-6

                n_co = 1.4630 #core refractive index
                n_cl = 1.4585 #cladding refractive index

                fiber_core_diameter = output_fiber_core_diameter # [m]
                a = fiber_core_diameter / 2 #fiber core radius

                #find the v parameter range that corresponds to the 1550 nm wavelength
                #v_1550 = 2 * np.pi * a / lda * np.sqrt(n_co**2 - n_cl**2)
                v_1550 = 4.5

                b = np.linspace(1e-3, 1-1e-3, 400000)
                V_list = np.linspace(2, 5.1, 20)
                V_obs = v_1550
                #b, V = np.meshgrid(b_list, V_list)

                b_solutions_at_V_obs = []
                b_solutions = []    
                for V in V_list:
                    LHS = np.sqrt(1-b) * jv(n+1, V * np.sqrt(1-b)) / jv(n, V * np.sqrt(1-b))
                    RHS = np.sqrt(b) * kv(n+1, V * np.sqrt(b)) / kv(n, V * np.sqrt(b))

                    intersection_index = np.argmax(np.where(np.isclose(LHS, RHS, atol=0.000001),1,0).flatten())

                    b_solutions.append(b[intersection_index])

                LHS = np.sqrt(1-b) * jv(n+1, V_obs * np.sqrt(1-b)) / jv(n, V_obs * np.sqrt(1-b))
                RHS = np.sqrt(b) * kv(n+1, V_obs * np.sqrt(b)) / kv(n, V_obs * np.sqrt(b))
                b_solutions_at_V_obs.append(b[np.argmax(np.where(np.isclose(LHS, RHS, atol=0.00001),1,0).flatten())])

                b_mode_sel = b_solutions_at_V_obs[0]

                k_co = 2 * np.pi / lda * n_co
                k_cl = 2 * np.pi / lda * n_cl

                beta = np.sqrt(b_mode_sel * (k_co**2 - k_cl**2) + k_cl**2)

                chi_co = v_1550 * np.sqrt(1 - b_mode_sel) / a
                chi_cl = v_1550 * np.sqrt(b_mode_sel) / a

                E_field = np.zeros_like(X, dtype=complex)

                for i in range(Nx):
                    for j in range(Ny):
                        r = np.sqrt(X[i,j]**2 + Y[i,j]**2)
                        if r < 10*a:
                            phi = np.arctan2(Y[i,j], X[i,j])
                            if r < a:#inside core
                                E_field[i,j] = jv(n,chi_co * r) / jv(n, chi_co * a) * np.cos(n * phi)
                            else:#outside core
                                E_field[i,j] = kv(n, chi_cl * r) / kv(n, chi_cl * a) * np.cos(n * phi)
                #save the computed pattern to a file for future use
                pattern_file = fiber_modes_input_folder + f"/LP{n}1_field_distribution_{Nx}.npy"
                np.save(pattern_file, E_field)
                output_patterns.append(E_field)
        return output_patterns
    except:
        raise(Exception("Error in computing fiber patterns"))

def shift_spatial_grid(shift_amount_x,shift_amount_y):
    '''Shifts the simulation spatial grid by a given physical (true space) amount. 
    Args:
        shift_amount_x: amount to shift the grid in x direction (in meters)
        shift_amount_y: amount to shift the grid in y direction (in meters)
    Returns:
        the shifted spatial grid (X, Y)'''
    Xp = X + shift_amount_x
    Yp = Y + shift_amount_y
    return Xp, Yp

def get_source_field(beam_waist,center_x=0,center_y=0,):
    '''Returns a 2D normalized Gaussian source field for a given beam waist and centered at the specified (x,y) coordinates. 
    Normalization consists in scaling the field such that its integrated intensity is 1.
    
    Args:
        beam_waist: the waist of the Gaussian beam (in meters)
        center_x: the x-coordinate of the center of the Gaussian beam (in meters)
        center_y: the y-coordinate of the center of the Gaussian beam (in meters)
    Returns:
        The 2D normalized Gaussian source field.
    '''

    if center_x != 0 or center_y != 0:
        X_shifted, Y_shifted = shift_spatial_grid(center_x, center_y)
        rho = np.sqrt(X_shifted**2 + Y_shifted**2)
    else:
        rho = np.sqrt(X**2 + Y**2)

    gaussian_field = np.exp(-(rho)**2 / (beam_waist)**2) #Gaussian input field, flattened
    gaussian_field_normalized = (gaussian_field / np.sqrt(np.sum(np.abs(gaussian_field)**2))) #normalize the input field to have power = 1
    
    return gaussian_field_normalized

def propagate_source(source_to_propagate):
    '''Returns the propagated source field (flattened) just before metasurface 1 (MS1). 
    This involves propagating the source for a distance d0: from its plane of definition up to MS1.
    Args:
        source_to_propagate: the source field to be propagated (flattened)
    Returns:
        The propagated source field (flattened) just before MS1.
    '''
    
    fft_input_array[:,:] = np.fft.ifftshift(source_to_propagate.reshape((Nx, Ny)))
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_0_nat
    ifft_operator()
    propagated_source_field = np.fft.fftshift(ifft_operator.output_array).flatten()
    
    return propagated_source_field

def propagate_source_before_MS1(source_to_propagate):
    '''Returns the propagated source field (flattened) just before metasurface 1 (MS1). 
    This involves propagating the source for a distance d0: from its plane of definition up to MS1.
    Args:
        source_to_propagate: the source field to be propagated (flattened)
    Returns:
        The propagated source field (flattened) just before MS1.
    '''
    
    fft_input_array[:,:] = np.fft.ifftshift(source_to_propagate.reshape((Nx, Ny)))
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_0_nat_source
    ifft_operator()
    propagated_source_field = np.fft.fftshift(ifft_operator.output_array).flatten()
    
    return propagated_source_field

def make_1Dplot_of(x_axis,y_axis,plot_zoom_x=zoom_x,save_name="plot_1D",) -> None:
    '''Makes a 1D plot of the given data and saves it to a PDF file.
    Args:
        x_axis: list of x-axis data (list of 1D arrays)
        y_axis: list of y-axis data (list of 1D arrays)
        plot_zoom_x: zoom level for the x-axis (in micrometers)
        save_name: name of the file to save the plot (without extension)
    Returns:
        None
    '''
    if len(x_axis) != len(y_axis):
        raise ValueError("Lists x_axis and y_axis must have the same length.")
    plt.figure()
    for i in range(len(x_axis)):
        plt.plot(x_axis[i], y_axis[i])
    plt.xlabel("x [um]")
    plt.tight_layout()
    plt.xlim(-plot_zoom_x, plot_zoom_x)
    plt.savefig(f"{save_folder}/" + save_name + ".pdf")
    plt.close()

def make_2Dplot_of(given_field,choose_quantity="amplitude",save_name="plot",plot_zoom_x=zoom_x,plot_zoom_y=zoom_y) -> None:
    '''Makes a plot of the given 2D field in a zoomed region. 
    Saves the plot with the chosen filename to a PDF file. 
    Args:
        given_field: the 2D field to be plotted (flattened)
        choose_quantity: either "amplitude" or "phase" to select quantity to plot
        save_name: name of the file to save the plot (without extension)
        plot_zoom_x: zoom level for the x-axis (in micrometers)
        plot_zoom_y: zoom level for the y-axis (in micrometers)
    Returns:
        None'''
    
    given_field_2d = given_field.reshape((Nx, Ny)) #assure the field is 2D before plotting
    plt.figure()
    if choose_quantity == "amplitude":
        plt.imshow(np.abs(given_field_2d), extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2))
    elif choose_quantity == "phase":
        plt.imshow((np.angle(given_field_2d)+2*np.pi)%(2*np.pi), extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2), cmap='hsv', vmin=0, vmax=2*np.pi)
    elif choose_quantity == "phase_mask":
        plt.imshow((given_field_2d+2*np.pi)%(2*np.pi), extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2), cmap='hsv', vmin=0, vmax=2*np.pi)
    else:
        raise ValueError("choose_quantity must be either 'amplitude' or 'phase'")

    plt.xlim(-plot_zoom_x, plot_zoom_x)
    plt.ylim(-plot_zoom_y, plot_zoom_y)
    plt.xlabel("x [um]")
    plt.ylabel("y [um]")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{save_folder}/" + save_name + ".pdf")
    plt.close()

def make_polarPlot_of(given_field,choose_quantity="amplitude",save_name="polar_plot",plot_zoom_r=zoom_x) -> None:
    '''Makes a polar plot of the given 2D field in a zoomed region. 
    Saves the plot with the chosen filename to a PDF file. 
    Args:
        given_field: the 2D field to be plotted (flattened)
        choose_quantity: either "amplitude" or "phase" to select quantity to plot
        save_name: name of the file to save the plot (without extension)
        plot_zoom_r: zoom level for the radial axis (in micrometers)
    Returns:
        None
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    if choose_quantity == "amplitude":
        given_field_2d = np.abs(given_field.reshape((Nx, Ny))) #assure the field is 2D before plotting
        im = ax.imshow(given_field_2d,origin='lower')
    elif choose_quantity == "phase":
        given_field_2d = (np.angle(given_field.reshape((Nx, Ny))) + 2*np.pi) % (2*np.pi)#assure the field is 2D before plotting
        im = ax.imshow(given_field_2d, cmap='hsv', vmin=0, vmax=2*np.pi)
    else:
        raise ValueError("choose_quantity must be either 'amplitude' or 'phase'")
    
    center_x, center_y = Nx//2, Ny//2
    radius = plot_zoom_r * 1e-6 / (size_x/2) * (Nx/2) #convert zoom in um to pixels
    circle = patches.Circle((center_x, center_y), radius, transform=ax.transData)
    im.set_clip_path(circle)
    ax.set_xlim(center_x - radius, center_x + radius)
    ax.set_ylim(center_y - radius, center_y + radius)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax)
    plt.savefig(f"{save_folder}/" + save_name + ".pdf")

def symmetrize(w):
    '''
    Takes an input array, which must be full in size, turns it into a 2D array, and symmetrizes it along the x and y axes.
    Returns the first quadrant of shape (nx,ny), including the zero axis, flattened
    '''
    try:
        w_2d = w.reshape((Nx, Ny))
    except:
        raise ValueError(f"Input array w cannot be reshaped to ({Nx}, {Ny}). Current shape: {w.shape}")

    w_symm = np.zeros((nx+1, ny+1))
    w_symm[:,:] = w_2d[:nx+1, :ny+1]

    return w_symm.flatten()

def expand_symmetrize(w_symm):
    '''
    Takes a array of shape (nx+1, ny+1), representing the first quadrant, including zero axes, and 
    expands it to a full 2D array of shape (Nx, Ny) by mirroring, flattened.
    '''

    w_symm_2d = w_symm.reshape((nx+1, ny+1))

    w_full = np.zeros((Nx, Ny))

    w_full[:nx+1, :ny+1] = w_symm_2d
    w_full[:nx+1, ny+1:] = np.flip(w_full[:nx+1, :ny], axis=1)
    w_full[nx+1:, :ny+1] = np.flip(w_full[:nx, :ny+1], axis=0)
    w_full[nx+1:, ny+1:] = np.flip(np.flip(w_full[:nx, :ny], axis=1), axis=0)
    w_full[nx, ny] = w_symm_2d[-1, -1]

    return w_full.flatten()

###########################
#initialize source field
source_field = get_source_field(beam_waist)
source_field_2d = source_field.reshape((Nx, Ny)) #reshape to 2D for plotting
#source field integrated intensity
source_power = np.sum(np.abs(source_field)**2)
print("Source field integrated intensity: %.4e" % source_power)
#make a plot and save the source field
make_2Dplot_of(source_field_2d, choose_quantity="amplitude", save_name="source_field_amplitude", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
make_2Dplot_of(source_field_2d, choose_quantity="phase", save_name="source_field_phase", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
###########################

###########################
#initialize input field (propagated source field)
input_field = propagate_source(source_field)
input_field_2d = input_field.reshape((Nx, Ny)) #reshape to 2D for plotting
#save the input field to numpy array
np.save(f"{save_folder}/input_field_{d0}_{d1}.npy", input_field_2d)
#propagate the source field to the plane just before MS1 and save it to a numpy array, to be used in FDTD simulations
input_field_before_MS1 = propagate_source_before_MS1(source_field)
np.save(f"{save_folder}/input_field_before_MS1.npy", input_field_before_MS1.reshape((Nx, Ny)))


#input field integrated intensity
input_field_intensity = np.sum(np.abs(input_field)**2)
print("Input field integrated intensity: %.4e" % input_field_intensity)
#make a plot and save the input field
make_2Dplot_of(input_field_2d, choose_quantity="amplitude", save_name="input_field_amplitude", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
make_2Dplot_of(input_field_2d, choose_quantity="phase", save_name="input_field_phase", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
make_2Dplot_of(input_field_before_MS1.reshape((Nx, Ny)), choose_quantity="amplitude", save_name="input_field_before_MS1_amplitude", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
make_2Dplot_of(input_field_before_MS1.reshape((Nx, Ny)), choose_quantity="phase", save_name="input_field_before_MS1_phase", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
###########################

###########################
#initialize target field
target_mode_indices = [2] 
fiber_mode_pattern = get_fiber_mode_pattern(target_mode_indices)
target_Efield_2d = fiber_mode_pattern / np.sqrt(np.sum(np.abs(fiber_mode_pattern)**2))
target_Efield =  target_Efield_2d.flatten() #normalized target field (intensity = 1)

# Plot: Target field
make_2Dplot_of(target_Efield_2d, choose_quantity="amplitude", save_name="target_field_1_amplitude_polar", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
make_2Dplot_of(target_Efield_2d, choose_quantity="phase", save_name="target_field_1_phase_polar", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
###########################

###########################
#initialize the normalized phase masks and global phase mask
w_norm = np.zeros((2*S)) #concatenated normalized parameters for both masks
rho = np.sqrt(X**2 + Y**2)
circular_mask_1 = np.where(rho.flatten() < output_fiber_core_diameter/3, 1, 0)
#w_norm[:S] += circular_mask_1
target_pattern_phase = (np.angle(target_Efield) + 2 * np.pi) % (2 * np.pi)*circular_mask_1 #make sure the phase is between 0 and 2pi
w_norm[S:] += target_pattern_phase / (2 * np.pi) #normalize the target phase to be between 0 and 1

w_norm_symm_1 = symmetrize(w_norm[:S])
w_norm_symm_2 = symmetrize(w_norm[S:])

w_norm_symm = np.concatenate((w_norm_symm_1, w_norm_symm_2), axis=0)

phase_mask = phase_given_w(w_norm)

#define bounding box for the optimization. The outer mask is fixed to zero phase, while the inner mask can vary between 0 and 2pi phase.
#the bounding box is the outer shell of a square with thickness 100 um
norm_phase_min = 0
norm_phase_max = 1
mask_buffer = 100e-6
#the upper bounds are set to 1 only within the fibre cladding diameter and zero outside
mask_upper_bounds = norm_phase_max * np.where( X.flatten()**2 + Y.flatten()**2 < (max_available_diameter/2)**2, 1, 0)
mask_upper_bounds_symm = mask_upper_bounds.reshape((Nx,Ny))[:nx+1, :ny+1].flatten()

mask_upper_bounds_symm = np.concatenate((mask_upper_bounds_symm, mask_upper_bounds_symm),axis=0)
mask_lower_bounds_symm = np.zeros_like(mask_upper_bounds_symm)
###########################

intermediate_fields = [None, None]
def forward_propagate() -> None:
    '''Propagates the input field through the system. Stores results in-place within the intermediate_fields list.
    Returns:
        None
    '''
    print("Starting propagating...")
    start_time = time.time()

    fft_input_array[:,:] = np.fft.ifftshift((input_field * np.exp(1j * phase_mask[:S])).reshape((Nx, Ny)))
    fft_operator()
    #print("BEFORE")
    fft_power = np.sum(np.abs(fft_operator.output_array)**2)/S
    non_propagating_mask = np.where((wavelength * nu_parallel)**2 > 1, 0, 1)
    non_propagating_mask = np.fft.ifftshift(non_propagating_mask)
    propagating_power = np.sum(np.abs(fft_operator.output_array * non_propagating_mask)**2)/S
    #print("FFT power: %.4e, Propagating power: %.4e, Non-propagating power: %.4e" % (fft_power, propagating_power, fft_power - propagating_power))
    ifft_input_array[:,:] = fft_operator.output_array * P_1_nat #element wise product
    #print("AFTER")

    fft_power = np.sum(np.abs(ifft_operator.input_array)**2)/S
    non_propagating_mask = np.where((wavelength * nu_parallel)**2 > 1, 0, 1)
    non_propagating_mask = np.fft.ifftshift(non_propagating_mask)
    propagating_power = np.sum(np.abs(ifft_operator.input_array * non_propagating_mask)**2)/S
    #print("FFT power: %.4e, Propagating power: %.4e, Non-propagating power: %.4e" % (fft_power, propagating_power, fft_power - propagating_power))
    
    ifft_operator()
    intermediate_fields[0] = np.fft.fftshift(ifft_operator.output_array.copy())
    print("Field intensity after first phase mask and propagation: %.4e" % np.sum(np.abs(intermediate_fields[0])**2))

    fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(ifft_operator.output_array) * np.exp(1j * phase_mask[S:].reshape((Nx, Ny))))
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_2_nat #element wise product
    ifft_operator()
    intermediate_fields[1] = np.fft.fftshift(ifft_operator.output_array.copy())
    
    end_time = time.time()
    make_2Dplot_of(intermediate_fields[0], choose_quantity="amplitude", save_name="intermediate_field_1_amplitude", plot_zoom_x=zoom_output_fiber_radius, plot_zoom_y=zoom_output_fiber_radius)
    make_2Dplot_of(intermediate_fields[1], choose_quantity="amplitude", save_name="intermediate_field_2_amplitude", plot_zoom_x=zoom_output_fiber_radius, plot_zoom_y=zoom_output_fiber_radius)
    
    print("Propagation finished in {:.6f} seconds.".format(end_time - start_time))


def reduce_symmetrized_gradient(g_full):
    g_2d = g_full.reshape((Nx, Ny))

    g_reduced = g_2d[:nx+1, :ny+1]
    third = g_2d[nx+1:,:ny+1]
    second = g_2d[:nx+1, ny+1:]
    fourth = g_2d[nx+1:, ny+1:]

    g_2d[:nx, ny+1:] += np.flip(fourth, axis = 0)
    g_2d[:nx, :ny+1] += np.flip(third, axis = 0)
    g_2d[:nx+1, :ny] += np.flip(second, axis = 1)
    
    return g_2d[:nx+1, :ny+1].flatten()

def adjoint_propagate():
    '''Backpropagates the output field using the adjoint of the propagation matrix P.
    Returns:
        the gradient of the cost function with respect to the phase mask (flattened)
    '''
    print("Computing adjoint...")
    start_time = time.time()

    Phi_1_dagger = np.exp(-1j * phase_mask[:S]).reshape((Nx, Ny))
    Phi_2_dagger = np.exp(-1j * phase_mask[S:]).reshape((Nx, Ny))

    #compute the adjoint source
    fft_input_array[:,:] = np.fft.ifftshift(np.sum(intermediate_fields[1] * np.conj(target_Efield_2d)) * target_Efield_2d)
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_2_dagger_nat
    ifft_operator()
    fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(ifft_operator.output_array) * Phi_2_dagger)

    grad_C_phi_2 = 2 * np.real(-1j * intermediate_fields[0].conj() * np.fft.fftshift(fft_input_array)).flatten() 

    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_1_dagger_nat #element wise
    ifft_operator()

    grad_C_phi_1 = 2 * np.real(-1j * input_field_2d.conj() * np.fft.fftshift(ifft_operator.output_array) * Phi_1_dagger).flatten()
    
    end_time = time.time()
    print("Adjoint finished in {:.6f} seconds.".format(end_time - start_time))
    return grad_C_phi_1, grad_C_phi_2

def _compute_cost():

    forward_propagate()

    output_field = intermediate_fields[-1].flatten()

    #circular_mask = np.where(rho.flatten() < 10*6e-6, 1, 0) 
    C_s = np.abs(np.sum(output_field * np.conj(target_Efield))) ** 2
    C_s = np.real(np.sum(C_s))

    C_t = C_s

    return C_t   

opt_history = []
iter_num = [0]
def cost_fun(x, grad):
    ''' Cost function to be minimized.
    Args:
        x: input normalized parameter field (flattened)
        grad: gradient of the cost function with respect to x, modified in place (flattened)
    Returns:
        The cost function value for the input parameter field x.
    '''
    #since the objective function is to maximize the overlap, and the target mode is mirror symmetric, 
    #we can let the algorithm optimize only a fourth of the phase mask and then synthetize automatically the remaining three quadrants.
    #Since we have two metasurfaces, the total number of parameters the algorithm needs to optimize is S/2 (S/4 for each metasurface)
    ms1_first_quadrant = x[:dof//2].reshape(nx+1, ny+1)
    ms2_first_quadrant = x[dof//2:].reshape(nx+1, ny+1)

    ms1 = expand_symmetrize(ms1_first_quadrant)
    ms2 = expand_symmetrize(ms2_first_quadrant)

    all_ms = np.concatenate((ms1, ms2))

    w_norm[:] = all_ms

    phase_mask[:] = phase_given_w(w_norm)


    C = _compute_cost() 

    opt_history.append(C)
    iter_num[0] += 1
    if iter_num[0] % 10 == 0:
        plt.figure()
        plt.plot(opt_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function Value')
        plt.title('Optimization History')
        plt.tight_layout()
        plt.savefig(f"{save_folder}/optimization_history.pdf")
        plt.close()

    if grad.size > 0:
        full_grad1, full_grad2 = adjoint_propagate()
        reduced_grad1 = reduce_symmetrized_gradient(full_grad1)
        reduced_grad2 = reduce_symmetrized_gradient(full_grad2)
        reduced_grads = np.concatenate((reduced_grad1, reduced_grad2), axis=0)
        grad[:] = reduced_grads * 2 * np.pi         
    return C

if __name__ == "__main__":
    #initialize nlopt solver
    solver = nlopt.opt(nlopt.LD_CCSAQ, dof)
    
    solver.set_lower_bounds(mask_lower_bounds_symm.flatten())
    solver.set_upper_bounds(mask_upper_bounds_symm.flatten())
    solver.set_max_objective(cost_fun)
    solver.set_maxeval(opt_max_eval)
    solver.set_param("dual_ftol_rel", 1e-7)
    solver.set_param("verbosity",1)

    print("Starting optimization...")

    start = True
    if start:
        w_norm_symm[:] = solver.optimize(w_norm_symm)
        w_norm[:S] = expand_symmetrize(w_norm_symm[:dof//2])
        w_norm[S:] = expand_symmetrize(w_norm_symm[dof//2:])

    print("Optimization completed.")
    print("LAST OPTIMUM VALUE: ", solver.last_optimum_value())
    #verify the results
    phase_mask = phase_given_w(w_norm)
    forward_propagate()
    field_input_ms2 = intermediate_fields[0]
    output_field = intermediate_fields[-1]
    
    input_power = np.sum(np.abs(input_field)**2)
    output_power = np.sum(np.abs(output_field)**2)

    print("Norm. input power: %.4e" % input_power)
    print("Norm. output power: %.4e" % output_power)

    #save the output field for later use
    save_output_field = True
    if save_output_field:
        np.save(f"{save_folder}/intermediate_field_1.npy", intermediate_fields[0].reshape((Nx, Ny)))
        np.save(f"{save_folder}/optimized_output_field.npy", output_field.reshape((Nx, Ny)))
    # Plot results in individual PDF figures
    
    # Reshape fields to 2D for visualization
    output_field_2d = output_field.reshape((Nx, Ny))
    phase_mask_1 = phase_given_w(w_norm[:S]).reshape((Nx, Ny))
    phase_mask_2 = phase_given_w(w_norm[S:]).reshape((Nx, Ny))

    phase_mask_zoom = 150
    make_2Dplot_of(phase_mask_1, choose_quantity="phase_mask", save_name="phase_mask_1", plot_zoom_x=phase_mask_zoom, plot_zoom_y=phase_mask_zoom)
    make_2Dplot_of(phase_mask_2, choose_quantity="phase_mask", save_name="phase_mask_2", plot_zoom_x=phase_mask_zoom, plot_zoom_y=phase_mask_zoom)

    # Plot: Output field amplitude and phase
    make_polarPlot_of(output_field_2d, choose_quantity="amplitude", save_name="optimized_output_field_amplitude", plot_zoom_r=zoom_output_fiber_radius)
    make_polarPlot_of(output_field_2d, choose_quantity="phase", save_name="optimized_output_field_phase", plot_zoom_r=zoom_output_fiber_radius)
    
    slice_y_index = Ny//2
    slice_x = xs * 1e6
    slice_target = target_Efield.reshape((Nx, Ny))[slice_y_index,:]
    slice_output = output_field.reshape((Nx, Ny))[slice_y_index,:]
    make_1Dplot_of([slice_x, slice_x], [np.abs(slice_target), np.abs(slice_output)], plot_zoom_x=1e6*output_fiber_core_diameter, save_name="modeconv_output_vs_target_amplitude")
    make_1Dplot_of([slice_x, slice_x], [(np.angle(slice_target)+2*np.pi)%(2*np.pi), (np.angle(slice_output)+2*np.pi)%(2*np.pi)], plot_zoom_x=1e6*output_fiber_core_diameter, save_name="modeconv_output_vs_target_phase")

    #save the phase masks for later use
    savePhase_masks = True
    if savePhase_masks:
        np.save(f"{save_folder}/optimized_phase_mask_1.npy", phase_mask_1)
        np.save(f"{save_folder}/optimized_phase_mask_2.npy", phase_mask_2)

    print(f"All plots saved to {save_folder}/ folder.")