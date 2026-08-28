import time
import multiprocessing

from mpi4py.futures import MPIPoolExecutor

import pyfftw
#pyFFTW setup
n_cpu = 1
pyfftw.config.NUM_THREADS = n_cpu
pyfftw.interfaces.cache.disable()
pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'  # Use ESTIMATE for deterministic behavior (no algorithm measurement)

import nlopt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#set the global font size for all plots
plt.rcParams.update({'font.size': 14})

from scipy.special import jv, kv

from smt.applications.ego import EGO, KRG
from smt.sampling_methods import LHS
from smt.design_space import (
    DesignSpace,
) 
import pickle
import os
results_folder = "results_symm_ego_testing"

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

d = np.zeros((3,))

phase_min = 0
phase_max = 2 * np.pi
##############################

##############################
#Source
beam_waist = 5.2e-6
#output fibre
output_fiber_core_diameter = 19e-6
#fibre cladding diameter
fibre_cladding_diameter = 125e-6 #[um]
max_available_diameter = 122e-6 #this is the metasurface maximum diameter (must be smaller than cladding)
##############################

##############################
#plan pyfftw objects for forward and inverse FFTs
fft_input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
fft_output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
ifft_input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
ifft_output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)

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

phase_factor_0 = k0 * d[0] * np.sqrt(1 - (lda0 * nu_parallel) ** 2 + 0*1j)
P_0 = np.exp(1j * phase_factor_0)
P_0_nat = np.fft.ifftshift(P_0)
phase_factor_1 = k1 * d[1] * np.sqrt(1 - (lda1 * nu_parallel) ** 2 + 0*1j)
P_1 = np.exp(1j * phase_factor_1)
P_1_nat = np.fft.ifftshift(P_1)
phase_factor_2 = k2 * d[2] * np.sqrt(1 - (lda2 * nu_parallel) ** 2 + 0*1j)
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

def get_fiber_mode_pattern(mode_list:list):
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
    plt.savefig(results_folder + "/" + save_name + ".pdf")
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
    plt.savefig(results_folder + "/" + save_name + ".pdf")
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
    plt.savefig(results_folder + "/" + save_name + ".pdf")

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
#print("Source field integrated intensity: %.4e" % source_power)
#make a plot and save the source field
#make_2Dplot_of(source_field_2d, choose_quantity="amplitude", save_name="source_field_amplitude", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
#make_2Dplot_of(source_field_2d, choose_quantity="phase", save_name="source_field_phase", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
###########################

###########################
#initialize input field (propagated source field)
input_field = propagate_source(source_field)
input_field_2d = input_field.reshape((Nx, Ny)) #reshape to 2D for plotting
#save the input field to numpy array
#np.save(f'{results_folder}/input_field_{d[0]}_{d[1]}.npy', input_field_2d)
#input field integrated intensity
input_field_intensity = np.sum(np.abs(input_field)**2)
#print("Input field integrated intensity: %.4e" % input_field_intensity)
#make a plot and save the input field
#make_2Dplot_of(input_field_2d, choose_quantity="amplitude", save_name="input_field_amplitude", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
#make_2Dplot_of(input_field_2d, choose_quantity="phase", save_name="input_field_phase", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
###########################

###########################
#initialize target field
target_mode_indices = [2] 
fiber_mode_pattern = get_fiber_mode_pattern(target_mode_indices)
target_Efield_2d = fiber_mode_pattern / np.sqrt(np.sum(np.abs(fiber_mode_pattern)**2))
target_Efield =  target_Efield_2d.flatten() #normalized target field (intensity = 1)

# Plot: Target field
#make_polarPlot_of(target_Efield_2d, choose_quantity="amplitude", save_name="target_field_1_amplitude_polar", plot_zoom_r=zoom_output_fiber_radius)
#make_polarPlot_of(target_Efield_2d, choose_quantity="phase", save_name="target_field_1_phase_polar", plot_zoom_r=zoom_output_fiber_radius)
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
    #print("Starting propagating...")
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
    #print("Field intensity after first phase mask and propagation: %.4e" % np.sum(np.abs(intermediate_fields[0])**2))

    fft_input_array[:,:] = np.fft.ifftshift(np.fft.fftshift(ifft_operator.output_array) * np.exp(1j * phase_mask[S:].reshape((Nx, Ny))))
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_2_nat #element wise product
    ifft_operator()
    intermediate_fields[1] = np.fft.fftshift(ifft_operator.output_array.copy())
    
    end_time = time.time()
    #make_2Dplot_of(intermediate_fields[0], choose_quantity="amplitude", save_name="intermediate_field_1_amplitude", plot_zoom_x=full_view_x, plot_zoom_y=full_view_y)
    #make_2Dplot_of(intermediate_fields[1], choose_quantity="amplitude", save_name="intermediate_field_2_amplitude", plot_zoom_x=full_view_x, plot_zoom_y=full_view_y)
    
    #print("Propagation finished in {:.6f} seconds.".format(end_time - start_time))

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
    #print("Computing adjoint...")
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
    #print("Adjoint finished in {:.6f} seconds.".format(end_time - start_time))
    return grad_C_phi_1, grad_C_phi_2

def _compute_cost():
    forward_propagate()

    output_field = intermediate_fields[-1].flatten()

    #circular_mask = np.where(rho.flatten() < 10*6e-6, 1, 0) 
    C_s = np.abs(np.sum(output_field * np.conj(target_Efield))) ** 2
    C_s = np.real(np.sum(C_s))

    C_t = C_s

    return C_t   

iter_num = [0]
opt_history = []
def cost_fun(x, grad):
    ''' Cost function to be minimized.
    Args:
        x: input normalized parameter field (flattened)
        grad: gradient of the cost function with respect to x, modified in place (flattened)
    Returns:
        The cost function value for the input parameter field x.
    '''
    ms1_first_quadrant = x[:dof//2].reshape(nx+1, ny+1)
    ms2_first_quadrant = x[dof//2:].reshape(nx+1, ny+1)

    ms1 = expand_symmetrize(ms1_first_quadrant)
    ms2 = expand_symmetrize(ms2_first_quadrant)

    all_ms = np.concatenate((ms1, ms2))

    w_norm[:] = all_ms

    phase_mask[:] = phase_given_w(w_norm)

    C = _compute_cost() 

    opt_history.append(C)

    if grad.size > 0:
        full_grad1, full_grad2 = adjoint_propagate()
        reduced_grad1 = reduce_symmetrized_gradient(full_grad1)
        reduced_grad2 = reduce_symmetrized_gradient(full_grad2)
        reduced_grads = np.concatenate((reduced_grad1, reduced_grad2), axis=0)
        grad[:] = reduced_grads * 2 * np.pi          
    return C

def metaprop():
    #initialize nlopt solver
    nlopt.srand(42)
    solver = nlopt.opt(nlopt.LD_CCSAQ, dof)
    
    solver.set_lower_bounds(mask_lower_bounds_symm.flatten())
    solver.set_upper_bounds(mask_upper_bounds_symm.flatten())
    solver.set_max_objective(cost_fun)
    solver.set_maxeval(opt_max_eval)
    solver.set_param("dual_ftol_rel", 1e-7)
    solver.set_param("verbosity",0)

    print("Starting optimization...")
    start = True
    if start:
        w_norm_symm[:] = solver.optimize(w_norm_symm)
        w_norm[:S] = expand_symmetrize(w_norm_symm[:dof//2])
        w_norm[S:] = expand_symmetrize(w_norm_symm[dof//2:])

    print("Optimization completed.")

    last_optimum_value = solver.last_optimum_value()
    print("Last optimum value: %.4e" % last_optimum_value)

    return last_optimum_value

def update_phase_factors():
    print("Phase factor distances: d0 = %.6f m, d1 = %.6f m, d2 = %.6f m" % (d[0], d[1], d[2]))
    phase_factor_0[:] = k0 * d[0] * np.sqrt(1 - (lda0 * nu_parallel) ** 2 + 0*1j)
    P_0[:] = np.exp(1j * phase_factor_0)
    P_0_nat[:] = np.fft.ifftshift(P_0)
    phase_factor_1[:] = k1 * d[1] * np.sqrt(1 - (lda1 * nu_parallel) ** 2 + 0*1j)
    P_1[:] = np.exp(1j * phase_factor_1)
    P_1_nat[:] = np.fft.ifftshift(P_1)
    phase_factor_2[:] = k2 * d[2] * np.sqrt(1 - (lda2 * nu_parallel) ** 2 + 0*1j)
    P_2[:] = np.exp(1j * phase_factor_2)
    P_2_nat[:] = np.fft.ifftshift(P_2)

    P_1_dagger[:] = P_1.T.conj()
    P_2_dagger[:] = P_2.T.conj()
    P_1_dagger_nat[:] = np.fft.ifftshift(P_1).T.conj()
    P_2_dagger_nat[:] = np.fft.ifftshift(P_2).T.conj()

def update_input_fields():
    input_field[:] = propagate_source(source_field)
    input_field_2d[:] = input_field.reshape((Nx, Ny)) #reshape to 2D for plotting
    #save the input field to numpy array
    #np.save("results/input_field.npy", input_field_2d)
    #input field integrated intensity
    #input_field_intensity = np.sum(np.abs(input_field)**2)
    #print("Input field integrated intensity: %.4e" % input_field_intensity)
    #make a plot and save the input field
    #make_2Dplot_of(input_field_2d, choose_quantity="amplitude", save_name="input_field_amplitude", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
    #make_2Dplot_of(input_field_2d, choose_quantity="phase", save_name="input_field_phase", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)

def initialize_phase_masks():
    #initialize the normalized phase masks and global phase mask
    w_norm[:] = np.zeros((2*S)) #concatenated normalized parameters for both masks
    rho = np.sqrt(X**2 + Y**2)
    circular_mask_1 = np.where(rho.flatten() < output_fiber_core_diameter/3, 1, 0)
    #w_norm[:S] += circular_mask_1
    target_pattern_phase = (np.angle(target_Efield) + 2 * np.pi) % (2 * np.pi)*circular_mask_1 #make sure the phase is between 0 and 2pi
    w_norm[S:] += target_pattern_phase / (2 * np.pi) #normalize the target phase to be between 0 and 1

    phase_mask[:] = phase_given_w(w_norm)
    
    w_norm_symm_1 = symmetrize(w_norm[:S])
    w_norm_symm_2 = symmetrize(w_norm[S:])
    w_norm_symm[:] = np.concatenate((w_norm_symm_1, w_norm_symm_2), axis=0)

def reset_globals():
    opt_history[:] = []
    w_norm[:] = np.zeros((2*S))
    w_norm_symm[:] = np.zeros((dof))
    phase_mask[:] = phase_given_w(w_norm)
    d[:] = np.zeros((3,))

    # 2. Hard zero the PyFFTW aligned memory buffers (including stride padding)
    fft_input_array.fill(0)
    fft_output_array.fill(0)
    ifft_input_array.fill(0)
    ifft_output_array.fill(0)


def metaprop_update(x):
    '''Objective function called from EGO loop. Given a set of input design parameters x,
    it updates the system and calls the inner adjoint-based optimization.
    Changing the system distance parameters requires updating the following terms:
    1) propagation matrices
    2) input field'''

    #if not os.path.exists(f'{results_folder}/metaprop_results.txt'):
    #    with open(f'{results_folder}/metaprop_results.txt', 'w') as f:
    #        f.write("d0 [m] d1 [m] d2 [m] ob1 ob2 t\n")
    reset_globals()
    try:
        if x.shape[1]>1:
            x = x.flatten()
    except:
        pass
    C = 0
    for j in range(len(x)):
        #d[j] = 1e-6*np.round(x[j]) #update the global propagation distances with the new design parameters
        #truncatge the x[j] value to 0 decimal places
        d[j] = 1e-6*np.round(x[j], 0) #update the global propagation distances with the new design parameters

    d[-1] = d[0]
    assert np.all(d), "Propagation distances must be real-positive."
    print(f"Current propagation distances: d0 = {d[0]:.6f} m, d1 = {d[1]:.6f} m, d2 = {d[2]:.6f} m")
    
    update_phase_factors() #update the phase factors for the new propagation distances
    update_input_fields()
    initialize_phase_masks()
    last_optimum_value = metaprop() #solve the inner optimization problem for the given design parameters
    C = -last_optimum_value #remember ego does minimization
    #save all the partial results for a specific set of design parameters x to a text file
    #in append mode, so that we can keep track of the optimization history
    with open(f'{results_folder}/ego_evaluations.txt', 'a') as f:
        f.write(f"{d[0]:.6e} {d[1]:.6e} {last_optimum_value:.6e}\n")

    #print("Current cost function value: ", opt_history[-1])
    if hot_start: 
        return d[0], d[1], C
    else:
        return C

from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.image as mpimg
import matplotlib.animation as animation
import copy

def EI(GP, points, f_min):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    args0 = (f_min - pred) / np.sqrt(var)
    args1 = (f_min - pred) * norm.cdf(args0)
    args2 = np.sqrt(var) * norm.pdf(args0)

    if var.size == 1 and var == 0.0:  # can be use only if one point is computed
        return 0.0

    ei = args1 + args2
    return ei[0,0]
hot_start = True
def run_ego(design_space, n_ego_iter, criterion, xdoe_train, ydoe_train, xlimits):
    if not hot_start:#cold start
        sm = KRG(design_space=design_space, n_start=25, print_global=False)
        ego = EGO(
            n_iter=n_ego_iter,
            criterion=criterion,
            xdoe=xdoe_train,
            ydoe=ydoe_train,
            surrogate=sm,
            n_start=25,
        )
        #note that x_data contains the original train set plus the EGO evaluated points
        x_opt, y_opt, ind_best, x_data, y_data = ego.optimize(fun=metaprop_update) 
        #save the optimization results to a text file
        np.savetxt(f'{results_folder}/ego_opt_results.txt', np.hstack((x_opt,y_opt)))
        return x_opt, y_opt, ind_best, x_data, y_data
    else:#hot start
        #load the best surrogate model that was generated after cross-validation
        sm = pickle.load(open(f'{results_folder}/best_surrogate_model.pkl', 'rb'))
        #load the associated fold training data
        doe_train = pickle.load(open(f'{results_folder}/best_surrogate_model_training_data.pkl', 'rb'))
        xdoe_train_loaded, ydoe_train_loaded = doe_train
        x_data = copy.deepcopy(xdoe_train_loaded)
        y_data = copy.deepcopy(ydoe_train_loaded)
        n_start = 25 #number of initial random points to sample for EGO refinement
        x_et_k_tested_history = []
        EI_history = []
        for k in range(n_ego_iter):
            d0_start = np.atleast_2d(xlimits[0][0] + np.random.rand(n_start)*(xlimits[0][1]-xlimits[0][0]))
            d1_start = np.atleast_2d(xlimits[1][0] + np.random.rand(n_start)*(xlimits[1][1]-xlimits[1][0]))
            x_start = np.hstack((d0_start.T, d1_start.T))
            f_min_k = np.min(y_data)
            opt_all = np.array(#maximizes the expected improvement in a region around x_st, an element of the start point
                [
                    minimize(lambda x: float(-EI(sm, np.atleast_2d(x), f_min_k)), x_st, method="SLSQP", bounds=xlimits)
                    for x_st in x_start
                ]
            )
            opt_success = opt_all[[opt_i["success"] for opt_i in opt_all]]
            obj_success = np.array([opt_i["fun"] for opt_i in opt_success])
            ind_min = np.argmin(obj_success)
            #save the minimum EI value for this iteration to the history
            EI_history.append(-obj_success[ind_min])
            opt = opt_success[ind_min]
            #select the non-optimal points for history purposes. Later, we will plot
            #all the non-optimal points with crosses to show the search process of the n_start points
            non_optimal_points = [opt_i["x"] for i, opt_i in enumerate(opt_all) if i != ind_min]
            x_et_k = opt["x"] #this is the coordinate of the most interesting point: the one that maximizes the EI
            d0, d1, y_et_k = metaprop_update(x_et_k) #evaluate the objective function at the new point
            x_et_k_tested = np.atleast_2d([d0,d1])
            x_et_k_tested_history.append(x_et_k_tested)

            d0_EI = np.linspace(xlimits[0][0], xlimits[0][1], 100)
            d1_EI = np.linspace(xlimits[1][0], xlimits[1][1], 100)
            d0_EI_grid, d1_EI_grid = np.meshgrid(d0_EI, d1_EI)
            EI_grid = np.zeros_like(d0_EI_grid)
            for i in range(d0_EI_grid.shape[0]):
                for j in range(d0_EI_grid.shape[1]):
                    EI_grid[i,j] = EI(sm, np.atleast_2d([d0_EI_grid[i,j], d1_EI_grid[i,j]]), f_min_k)
            plt.figure(figsize=(8,6))
            plt.contourf(d0_EI_grid, d1_EI_grid, EI_grid, levels=50, cmap='viridis')
            plt.colorbar(label='Expected Improvement')
            #scatter the full history of tested points up to the current iteration k, the current point is red, the others are blue
            x_et_k_tested_history_array = np.vstack(x_et_k_tested_history)*1e6
            #scatter last point in red, the others in blue
            plt.scatter(x_et_k_tested_history_array[:-1,0], x_et_k_tested_history_array[:-1,1], color='blue', label='Previous best')
            plt.scatter(x_et_k_tested_history_array[-1,0], x_et_k_tested_history_array[-1,1], color='red', label='Best point')
            #also plot the non optimal points with crosses for this iteration
            if len(non_optimal_points) > 0:
                non_optimal_points_array = np.vstack(non_optimal_points)
                plt.scatter(non_optimal_points_array[:,0], non_optimal_points_array[:,1], color='orange', marker='x', label='Candidate')
            plt.title(f'EGO Iteration {k+1}/{n_ego_iter}')
            plt.xlabel('d0 [um]')
            plt.ylabel('d1 [um]')
            plt.xlim(xlimits[0])
            plt.ylim(xlimits[1])
            plt.legend(loc='upper left', framealpha=0.5)
            plt.tight_layout()
            plt.savefig(f'{results_folder}/Optimisation_{k}.png')
            plt.close()

            y_data = np.atleast_2d(np.append(y_data, y_et_k)).reshape(-1,1)
            x_data = np.atleast_2d(np.append(x_data, x_et_k_tested*1e6, axis=0))
            sm.set_training_values(x_data, y_data)
            sm.train() #train the model with the new data point
            print(f"{k}-th EGO refinement complete")
            #for this k iteration out of the n_start, make a plot of the EI surface and append the poisition of the 
            #selected point. Save the figure to a plot list ims, which will be used to create an animation at the end of the EGO loop
            #To test the EI surface, use a custom grid of d0,d1 points, and compute the EI for each point in the grid. Then plot the EI surface as a contour plot, and mark the selected point with a red dot.
            
        ind_best = np.argmin(y_data)
        x_opt = x_data[ind_best]
        y_opt = y_data[ind_best]
        np.savetxt(f'{results_folder}/ego_opt_results.txt', np.hstack((x_opt,y_opt)))
        ims = []
        fig = plt.figure(figsize=[10, 10])

        ax = plt.gca()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        for k in range(n_ego_iter):
            image_pt = mpimg.imread(f"{results_folder}/Optimisation_{k}.png")
            im = plt.imshow(image_pt)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=500)
        #save the animation to a gif file
        ani.save(f'{results_folder}/ego_optimization.gif', writer='pillow')

        #plot of the EI history
        plt.figure(figsize=(8,6))
        plt.plot(range(1, n_ego_iter+1), EI_history, marker='o')
        plt.title('Expected Improvement History')
        plt.xlabel('EGO Iteration')
        plt.ylabel('Expected Improvement')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{results_folder}/EI_history.pdf')
        plt.close()

        #surface plot of the refined surrogate model
        d0_grid = np.linspace(xlimits[0][0], xlimits[0][1], 100)
        d1_grid = np.linspace(xlimits[1][0], xlimits[1][1], 100)
        d0_grid_mesh, d1_grid_mesh = np.meshgrid(d0_grid, d1_grid)
        d_grid = np.column_stack((d0_grid_mesh.flatten(), d1_grid_mesh.flatten()))
        sm_predictions = sm.predict_values(d_grid).reshape(d0_grid_mesh.shape)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(d0_grid_mesh, d1_grid_mesh, -sm_predictions, cmap='viridis', alpha=0.7, edgecolor='none')
        ax.scatter(xdoe_train_loaded[:,0], xdoe_train_loaded[:,1], -ydoe_train_loaded.flatten(), color='blue', label='DOE Points')
        ax.scatter(x_data[len(xdoe_train_loaded):,0], x_data[len(xdoe_train_loaded):,1], -y_data[len(xdoe_train_loaded):].flatten(), color='red', label='EGO Evaluated Points')
        ax.scatter([x_opt[0]], [x_opt[1]], [-y_opt], color='green', s=100, label='Best Point', edgecolor='black')
        ax.set_xlabel('d0 [um]')
        ax.set_ylabel('d1 [um]')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{results_folder}/refined_surrogate_model_surface.pdf')
        plt.close()
        #contour plot
        plt.figure(figsize=(8,6))
        cp = plt.contourf(d0_grid_mesh, d1_grid_mesh, -sm_predictions, levels=50, cmap='viridis')
        plt.colorbar(cp, label='Surrogate Model Prediction')
        plt.scatter(xdoe_train_loaded[:,0], xdoe_train_loaded[:,1], color='blue', label='DOE Points')
        plt.scatter(x_data[len(xdoe_train_loaded):,0], x_data[len(xdoe_train_loaded):,1], color='red', label='EGO Evaluated Points')
        plt.scatter([x_opt[0]], [x_opt[1]], color='green', s=100, label='Best Point', edgecolor='black')
        plt.xlabel('d0 [um]')
        plt.ylabel('d1 [um]')
        plt.xlim(xlimits[0])
        plt.ylim(xlimits[1])
        plt.legend(framealpha=0.5)
        plt.tight_layout()
        plt.savefig(f'{results_folder}/refined_surrogate_model_contour.pdf')
        plt.close()

        pickle.dump(sm, open(f'{results_folder}/best_refined_surrogate_model.pkl', 'wb'))
        pickle.dump((x_data, y_data), open(f'{results_folder}/best_refined_surrogate_model_training_data.pkl', 'wb'))
        return x_opt, y_opt, ind_best, x_data, y_data

def validate_surrogate_model(design_space, xdoe_train, ydoe_train, xdoe_test, ydoe_test, validation_save_name, save_folder="results_symm_ego"):
    '''Validates the surrogate model (Kriging) by evaluating it on an unseen test set.
    Computes R² score and creates diagnostic plots.
    
    Args:
        design_space: SMT DesignSpace object
        xdoe_train: training input samples (N_train, n_dims)
        ydoe_train: training output values (N_train, 1)
        xdoe_test: test input samples (N_test, n_dims)
        ydoe_test: test output values (N_test, 1)
        save_folder: folder to save plots
    Returns:
        Dictionary with metrics (r2_score, rmse, mae, sm)
    '''
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Train the surrogate model
    print("\n" + "="*60)
    print("SURROGATE MODEL VALIDATION")
    print("="*60)
    print(f"Training set size: {xdoe_train.shape[0]} samples")
    print(f"Test set size: {xdoe_test.shape[0]} samples")
    
    sm = KRG(design_space=design_space, n_start=25, print_global=False)
    sm.set_training_values(xdoe_train, ydoe_train)
    sm.train()
    
    # Predict on both sets
    y_train_pred = sm.predict_values(xdoe_train)
    y_test_pred = sm.predict_values(xdoe_test)
    
    # Compute metrics
    r2_train = r2_score(ydoe_train, y_train_pred)
    r2_test = r2_score(ydoe_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(ydoe_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(ydoe_test, y_test_pred))
    mae_train = mean_absolute_error(ydoe_train, y_train_pred)
    mae_test = mean_absolute_error(ydoe_test, y_test_pred)
    
    print(f"\nTraining Set Metrics:")
    print(f"  R² Score: {r2_train:.4f}")
    print(f"  RMSE:     {rmse_train:.6e}")
    print(f"  MAE:      {mae_train:.6e}")
    
    print(f"\nTest Set Metrics:")
    print(f"  R² Score: {r2_test:.4f}")
    print(f"  RMSE:     {rmse_test:.6e}")
    print(f"  MAE:      {mae_test:.6e}")
    
    if r2_test < 0.80:
        print("\n  WARNING: Test R² < 0.80. Consider adding more DOE samples.")
    elif r2_test < 0.85:
        print("\n  CAUTION: Test R² < 0.85. Model quality is moderate.")
    else:
        print("\n Model quality is good. Ready to proceed with EGO.")
    
    print("="*60 + "\n")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Plot 1: Predicted vs Actual (Training and Test)
    ax = axes[0, 0]
    ax.scatter(ydoe_train, y_train_pred, alpha=0.6, label='Training', s=50)
    ax.scatter(ydoe_test, y_test_pred, alpha=0.6, label='Test', s=50, marker='^')
    y_min, y_max = min(ydoe_train.min(), ydoe_test.min()), max(ydoe_train.max(), ydoe_test.max())
    ax.plot([y_min, y_max], [y_min, y_max], 'k--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Output')
    ax.set_ylabel('Predicted Output')
    ax.set_title('Predicted vs Actual Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals (Test Set)
    ax = axes[0, 1]
    residuals = ydoe_test.flatten() - y_test_pred.flatten()
    ax.scatter(y_test_pred, residuals, alpha=0.6, s=50)
    ax.axhline(y=0, color='k', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Output')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Predicted Values (Test Set)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Design Space Coverage
    ax = axes[1, 0]
    ax.scatter(xdoe_train[:, 0], xdoe_train[:, 1], alpha=0.6, s=50, label='Training')
    ax.scatter(xdoe_test[:, 0], xdoe_test[:, 1], alpha=0.6, s=50, marker='^', label='Test')
    ax.set_xlabel('d0 [nm]')
    ax.set_ylabel('d1 [nm]')
    ax.set_title('Design Space Coverage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: R² Score Comparison
    ax = axes[1, 1]
    models = ['Training', 'Test']
    r2_scores = [r2_train, r2_test]
    colors = ['green' if r2 > 0.85 else 'orange' if r2 > 0.80 else 'red' for r2 in r2_scores]
    bars = ax.bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=0.85, color='green', linestyle='--', linewidth=2, label='Good (0.85)')
    ax.axhline(y=0.80, color='orange', linestyle='--', linewidth=2, label='Acceptable (0.80)')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.95, hspace=0.35, wspace=0.3)
    plt.savefig(f'{save_folder}/surrogate_model_validation_{validation_save_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'rmse_train': rmse_train,
        'rmse_test': rmse_test,
        'mae_train': mae_train,
        'mae_test': mae_test,
        'surrogate_model': sm
    }

if __name__ == "__main__":
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    ##############################
    #EGO setup
    ##############################
    n_ego_iter = 20
    xlimits = np.array([[100,700], [100,700]])
    seed = 777
    np.random.seed(seed)
    design_space = DesignSpace(xlimits, seed=seed)
    criterion = 'EI'
    ndoe = 40 #split into 70% train (28 samples) and 30% test (12 samples)
    #build DOE
    #sampling = LHS(xlimits=xlimits, seed=seed)
    #xdoe = sampling(ndoe)
    #ydoe = []

    #for xd in xdoe:
    #    ydoe.append(metaprop_update(xd))
    #ydoe = np.array(ydoe).reshape(-1,1)
    #with MPIPoolExecutor() as executor:
    #    ydoe = np.array(list(executor.map(metaprop_update, xdoe)))
    #    ydoe = ydoe.reshape(-1,1)

    #stacked = np.hstack((xdoe,ydoe))
    #save the doe results to a text file, building a dataset: each row has 4 columns: d0, d1, d2, cost function value
    #np.savetxt(f'{results_folder}/ego_doe.txt', stacked)

    #ydoe is the result of evaluating the xdoe samples
    #instead of giving the surrogate model the whole doe, we shall
    #define a train-test split of the doe, and give the surrogate model only the training set
    #train_size = int(ndoe * 0.7)
    #xdoe_train = xdoe[:train_size].reshape(-1,2)
    #ydoe_train = ydoe[:train_size].reshape(-1,1)
    #xdoe_test = xdoe[train_size:].reshape(-1,2)
    #ydoe_test = ydoe[train_size:].reshape(-1,1)

    #validate the surrogate model
    #validation_results = validate_surrogate_model(design_space, xdoe_train, ydoe_train, xdoe_test, ydoe_test, 'validation_before_ego', save_folder=results_folder)

    print("Starting EGO optimization...")

    x_opt, y_opt, ind_best, x_data, y_data = run_ego(design_space, n_ego_iter, criterion, 0, 0, xlimits)

    #after running ego, we need to make the following checks:
    #1) assemble the new dataset comprised of the original training points together with the new points evaluated by EGO,
    #2) the test points defined at the beginning shall remain untouched
    #3) we retrain the KRG surrogate using the original train set together with the new EGO evaluated points
    #4) we validate the surrogate model using the original test set and observe model performance metrics
    #5) we may want to visualize the 3D curve representing the model on the design space, 
    # plotting in the same figure the train points, the test points, the EGO evaluated points and the optimal point found by EGO
    #train_and_ego_points = x_data
    #train_and_ego_values = y_data
    #validation_results_after_optimization = validate_surrogate_model(design_space, train_and_ego_points, train_and_ego_values, xdoe_test, ydoe_test, 'validation_after_ego', save_folder=results_folder)
    #make the 3D plot of the surrogate model after EGO optimization
    #sm_after_ego = validation_results_after_optimization['surrogate_model']
    #fig = plt.figure(figsize=(10, 8))
    #ax = fig.add_subplot(111, projection='3d')
    # Create a grid for plotting the surrogate model surface
    # predictions are very cheap thanks to the model)
    #x1 = np.linspace(xlimits[0, 0], xlimits[0, 1], 50)
    #x2 = np.linspace(xlimits[1, 0], xlimits[1, 1], 50)
    #X1, X2 = np.meshgrid(x1, x2)
    #X_grid = np.column_stack((X1.flatten(), X2.flatten()))
    #y_grid_pred = sm_after_ego.predict_values(X_grid).reshape(X1.shape)
    #surf = ax.plot_surface(X1, X2, y_grid_pred, cmap='viridis', alpha=0.7, edgecolor='none')
    # Plot training points
    #ax.scatter(train_and_ego_points[:, 0], train_and_ego_points[:, 1], train_and_ego_values.flatten(), color='blue', marker='o', label='EGO Points', s=50)
    # Plot only the original training points (without the EGO additional)
    #ax.scatter(xdoe_train[:, 0], xdoe_train[:, 1], ydoe_train.flatten(), color='green', label='Original Train Points', s=50, marker='o')
    # Plot test points
    #ax.scatter(xdoe_test[:, 0], xdoe_test[:, 1], ydoe_test.flatten(), color='red', label='Test Points', s=50, marker='^')
    # Highlight the optimal point found by EGO
    #ax.scatter(x_opt[0], x_opt[1], y_opt, color='gold', label='EGO Optimum', s=100, edgecolor='black', marker='*')
    #ax.set_xlabel('d0 [nm]')
    #ax.set_ylabel('d1 [nm]')
    #ax.set_zlabel('Cost Function Value')
    #ax.legend()
    #plt.savefig(f'{results_folder}/surrogate_model_surface_after_ego.pdf', dpi=300, bbox_inches='tight')
    #plt.close()