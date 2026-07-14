import time
import multiprocessing

import pyfftw
#pyFFTW setup
n_cpu = multiprocessing.cpu_count()
pyfftw.config.NUM_THREADS = n_cpu
pyfftw.interfaces.cache.enable()
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

import nlopt

import numpy as np
import matplotlib.pyplot as plt
#set the global font size for all plots
plt.rcParams.update({'font.size': 14})

from scipy.special import jv, kv

##############################
#physics and simulation domain
wavelength = 1.55e-6
k0 = 2 * np.pi / wavelength

opt_max_eval = 250

unit_cell_pitch = 400e-9 #equivalent to spatial sampling

Nx = 2048 #pixels per dimension
Ny = Nx
S = Nx * Ny

size_x = unit_cell_pitch * Nx #actual physical size
size_y = unit_cell_pitch * Ny

xs = np.linspace(-size_x/2, size_x/2 - size_x / Nx, Nx)
ys = np.linspace(-size_y/2, size_y/2 - size_y / Ny, Ny)

X, Y = np.meshgrid(xs, ys)
rho = np.sqrt(X**2 + Y**2)

sampling_period = xs[1] - xs[0]

d0 = 300e-6 #propagation distance: source - MS1
d1 = 500e-6 #propagation distance: MS1 - MS2
d2 = 300e-6 #propagation distance: MS2 - target
d = [d1, d2] #d = [d1,d2] d1:distance MS1-MS2, d2: distance MS2-target

phase_min = 0
phase_max = 2 * np.pi
##############################

##############################
#Source
beam_waist = 4e-6
##############################

##############################
#plan pyfftw objects for forward and inverse FFTs
fft_input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
fft_output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
ifft_input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
ifft_output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)

fft_operator = pyfftw.FFTW(fft_input_array, fft_output_array, axes=(0,1), direction='FFTW_FORWARD', threads=n_cpu, flags=('FFTW_MEASURE','FFTW_DESTROY_INPUT',))
ifft_operator = pyfftw.FFTW(ifft_input_array, ifft_output_array, axes=(0,1), direction='FFTW_BACKWARD', threads=n_cpu, flags=('FFTW_MEASURE','FFTW_DESTROY_INPUT',))
##############################

##############################
#setting up spatial frequencies
ks = 2 * np.pi / sampling_period
kappas = np.arange(-Nx//2, Nx//2) * ks / Nx
KX, KY = np.meshgrid(kappas, kappas)
K_parallel = np.sqrt(KX**2 + KY**2)
nu_parallel = K_parallel / (2 * np.pi)

phase_factor_0 = k0 * d0 * np.sqrt(1 - (wavelength * nu_parallel) ** 2 + 0*1j)
P_0 = np.exp(1j * phase_factor_0)
P_0_nat = np.fft.ifftshift(P_0)
phase_factor_1 = k0 * d1 * np.sqrt(1 - (wavelength * nu_parallel) ** 2 + 0*1j)
P_1 = np.exp(1j * phase_factor_1)
P_1_nat = np.fft.ifftshift(P_1)
phase_factor_2 = k0 * d2 * np.sqrt(1 - (wavelength * nu_parallel) ** 2 + 0*1j)
P_2 = np.exp(1j * phase_factor_2)
P_2_nat = np.fft.ifftshift(P_2)

P_1_dagger = P_1.T.conj()
P_2_dagger = P_2.T.conj()
P_2_conj = P_2.conj()
P_1_dagger_nat = np.fft.ifftshift(P_1).T.conj()
P_2_dagger_nat = np.fft.ifftshift(P_2).T.conj()
##############################

##############################
#Plot zoom parameters [um]
zoom_x = 160
zoom_y = 160
all_view_x = size_x * 1e6 / 2
all_view_y = size_y * 1e6 / 2
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

def get_fiber_mode_pattern():
    '''Returns the target fiber mode pattern'''
    print("Computing target fiber mode pattern...")
    lda = 1.55e-6

    n_co = 1.4630
    n_cl = 1.4585
    a = 6e-6

    #find the v parameter range that corresponds to the 1550 nm wavelength
    #v_1550 = 2 * np.pi * a / lda * np.sqrt(n_co**2 - n_cl**2)
    v_1550 = 4.5

    b = np.linspace(1e-3, 1-1e-3, 400000)
    V_list = np.linspace(2, 5.1, 20)
    V_obs = v_1550
    #b, V = np.meshgrid(b_list, V_list)
    n_list = [0,1,2]

    b_solutions_at_V_obs = []
    for n in n_list:
        b_solutions = []    
        for V in V_list:
            LHS = np.sqrt(1-b) * jv(n+1, V * np.sqrt(1-b)) / jv(n, V * np.sqrt(1-b))
            RHS = np.sqrt(b) * kv(n+1, V * np.sqrt(b)) / kv(n, V * np.sqrt(b))
            

            intersection_index = np.argmax(np.where(np.isclose(LHS, RHS, atol=0.000001),1,0).flatten())

            b_solutions.append(b[intersection_index])

        LHS = np.sqrt(1-b) * jv(n+1, V_obs * np.sqrt(1-b)) / jv(n, V_obs * np.sqrt(1-b))
        RHS = np.sqrt(b) * kv(n+1, V_obs * np.sqrt(b)) / kv(n, V_obs * np.sqrt(b))
        b_solutions_at_V_obs.append(b[np.argmax(np.where(np.isclose(LHS, RHS, atol=0.00001),1,0).flatten())])

    n_mode_sel = 2
    b_mode_sel = b_solutions_at_V_obs[n_mode_sel]

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
                    E_field[i,j] = jv(n_mode_sel,chi_co * r) / jv(n_mode_sel, chi_co * a) * np.cos(n_mode_sel * phi)
                else:#outside core
                    E_field[i,j] = kv(n_mode_sel, chi_cl * r) / kv(n_mode_sel, chi_cl * a) * np.cos(n_mode_sel * phi)

    return E_field

def shift_spatial_grid(shift_amount_x,shift_amount_y):
    '''Shifts the spatial grid by a given amount. This is useful for centering the source field on the metasurface.
    Args:
        shift_amount_x: amount to shift the grid in x direction (in meters)
        shift_amount_y: amount to shift the grid in y direction (in meters)
    Returns:
        the shifted spatial grid (X, Y)'''
    Xp = X - shift_amount_x
    Yp = Y - shift_amount_y
    return Xp, Yp

def get_source_field(beam_waist,center_x=0,center_y=0,):
    '''Returns a normalized Gaussian source field for a given beam waist. Normalization consists in scaling
    the field such that its integrated intensity is 1.
    Args:
        beam_waist: the waist of the Gaussian beam (in meters)
        center_x: the x-coordinate of the center of the Gaussian beam (in meters)
        center_y: the y-coordinate of the center of the Gaussian beam (in meters)
    Returns:
        The normalized Gaussian source field
    '''

    if center_x != 0 or center_y != 0:
        X_shifted, Y_shifted = shift_spatial_grid(center_x, center_y)
        rho = np.sqrt(X_shifted**2 + Y_shifted**2)
    else:
        rho = np.sqrt(X**2 + Y**2)

    gaussian_field = np.exp(-(rho)**2 / (beam_waist)**2) #Gaussian input field, flattened
    gaussian_field_normalized = (gaussian_field / np.sqrt(np.sum(np.abs(gaussian_field)**2))) #normalize the input field to have power = 1
    return gaussian_field_normalized

def propagate_source():
    '''Returns the propagated source field just before metasurface 1 (MS1). This involves propagating the 
    source for a distance d0: from its plane of definition up to MS1.
    Returns:
        The propagated source field just before MS1 (flattened)
    '''
    fft_input_array[:,:] = source_field_2d
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_0_nat
    ifft_operator()
    propagated_source_field = ifft_operator.output_array.flatten()
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
    plt.savefig("results/" + save_name + ".pdf")
    plt.xlim(-plot_zoom_x, plot_zoom_x)
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
    plt.savefig("results/" + save_name + ".pdf")
    plt.close()

###########################
#initialize source field
source_spacing = 120e-6
source_field = get_source_field(beam_waist,center_x=source_spacing/2,center_y=0)
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
input_field = propagate_source()
input_field_2d = input_field.reshape((Nx, Ny)) #reshape to 2D for plotting
#input field integrated intensity
input_field_intensity = np.sum(np.abs(input_field)**2)
print("Input field integrated intensity: %.4e" % input_field_intensity)
#make a plot and save the input field
make_2Dplot_of(input_field_2d, choose_quantity="amplitude", save_name="input_field_amplitude", plot_zoom_x=zoom_x*2, plot_zoom_y=zoom_y*2)
make_2Dplot_of(input_field_2d, choose_quantity="phase", save_name="input_field_phase", plot_zoom_x=zoom_x*2, plot_zoom_y=zoom_y*2)
###########################

###########################
#initialize target field
fiber_mode_pattern = get_fiber_mode_pattern()
target_Efield_2d = fiber_mode_pattern / np.sqrt(np.sum(np.abs(fiber_mode_pattern)**2))
#shift the target field to math the source position
#target_Efield_2d = np.roll(target_Efield_2d, shift=int((source_spacing/2/unit_cell_pitch)), axis=1) #shift the target field to match the source position
target_Efield =  target_Efield_2d.flatten() #normalized target field (intensity = 1)

# Plot: Target field
make_2Dplot_of(target_Efield_2d, choose_quantity="amplitude", save_name="target_field_amplitude", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
make_2Dplot_of(target_Efield_2d, choose_quantity="phase", save_name="target_field_phase", plot_zoom_x=zoom_x, plot_zoom_y=zoom_y)
###########################

###########################
#initialize the normalized phase masks and global phase mask
w_norm = np.zeros((2*S)) #concatenated normalized parameters for both masks
X, Y = np.meshgrid(np.linspace(-size_x/2, size_x/2, Nx), np.linspace(-size_y/2, size_y/2, Ny))
rho = np.sqrt(X**2 + Y**2)

norm_phase_min = 0
norm_phase_max = 1
mask_buffer = 100e-6
mask_upper_bounds = norm_phase_max * np.where((np.abs(X.flatten()) > size_x/2 - mask_buffer) | (np.abs(Y.flatten()) > size_y/2 - mask_buffer), 0, 1)
mask_upper_bounds = np.concatenate((mask_upper_bounds, mask_upper_bounds),axis=0)
mask_lower_bounds = np.zeros_like(mask_upper_bounds)

def backward_propagate(field_to_backpropagate)-> np.ndarray:
    ''' Backpropagates the target field up to the second metasurface.
    Args:
        field_to_backpropagate: the field to be backpropagated (flattened)
    Returns:
        The backpropagated field at the second metasurface (flattened)
    '''
    fft_input_array[:,:] = field_to_backpropagate.reshape((Nx, Ny))
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_2_dagger_nat
    ifft_operator()
    backpropagated_field = ifft_operator.output_array.flatten()

    return backpropagated_field

def initialize_phase_masks():
    '''Initializes the normalized weights, in-place.'''
    initialize = True

    if initialize:
        #MS1
        #the first metasurface should steer light towards the center of the second metasurface
        #steer_angle = np.atan2(source_spacing/2,d1)
        steer_angle = 0.0001 
        grating_period = wavelength / np.sin(steer_angle)
        grating_phase = (2*np.pi - (2 * np.pi / grating_period) * X) % (2 * np.pi) #blazed grating phase
        w_norm[:S] = grating_phase.flatten() / (2 * np.pi) #normalized grating phase
        
        #MS2
        #propagate the source up to just before the second metasurface
        field_before_MS1 = input_field
        field_after_MS1 = field_before_MS1 * np.exp(1j * phase_given_w(w_norm[:S])) #apply the first phase mask
        fft_input_array[:,:] = field_after_MS1.reshape((Nx, Ny))
        fft_operator()
        ifft_input_array[:,:] = fft_operator.output_array * P_1_nat #element wise product
        ifft_operator()
        field_before_MS2 = ifft_operator.output_array.flatten() #field just before the second metasurface
        #make a plot of the field before the second metasurface
        make_2Dplot_of(field_before_MS2, choose_quantity="amplitude", save_name="field_before_MS2_amplitude", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
        make_2Dplot_of(field_before_MS2, choose_quantity="phase", save_name="field_before_MS2_phase", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)

        backpropagated_target_field = backward_propagate(target_Efield)
        #make plots of the backpropagated target field
        make_2Dplot_of(backpropagated_target_field, choose_quantity="amplitude", save_name="backpropagated_target_field_amplitude", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
        make_2Dplot_of(backpropagated_target_field, choose_quantity="phase", save_name="backpropagated_target_field_phase", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
        
        #MS2_phase = (np.angle(backpropagated_target_field) + 2 * np.pi) % (2 * np.pi) - (np.angle(field_before_MS2) + 2 * np.pi) % (2 * np.pi)
        #MS2_phase = (np.angle(target_Efield) + 2 * np.pi) % (2 * np.pi) - (np.angle(field_before_MS2) + 2 * np.pi) % (2 * np.pi)
        MS2_phase = (np.angle(target_Efield) + 2 * np.pi) % (2 * np.pi)
        MS2_phase = (MS2_phase + 2 * np.pi) % (2 * np.pi) #make sure the phase is between 0 and 2pi
        #normalize phase masks
        circular_mask = np.where(rho.flatten() < 20*6e-6, 1, 0) 
        w_norm[:S] = 0*circular_mask * grating_phase.flatten() / (2 * np.pi) 
        w_norm[S:] = circular_mask * MS2_phase.flatten() / (2 * np.pi)

#target_pattern_phase = (np.angle(target_Efield_2d) + 2 * np.pi) % (2 * np.pi) #make sure the phase is between 0 and 2pi
#target_pattern_phase = np.roll(target_pattern_phase, shift=int((source_spacing/2/unit_cell_pitch)), axis=1) #shift the target phase to match the source position

#initialize phase masks
initialize_phase_masks()
phase_mask = phase_given_w(w_norm)
make_2Dplot_of(phase_mask[:S].reshape((Nx, Ny)), choose_quantity="phase_mask", save_name="initial_phase_mask_1", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
make_2Dplot_of(phase_mask[S:].reshape((Nx, Ny)), choose_quantity="phase_mask", save_name="initial_phase_mask_2", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
#define bounding box for the optimization. The outer mask is fixed to zero phase, while the inner mask can vary between 0 and 2pi phase.
#the bounding box is the outer shell of a square with thickness 100 um

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
    print("BEFORE")
    fft_power = np.sum(np.abs(fft_operator.output_array)**2)/S
    non_propagating_mask = np.where((wavelength * nu_parallel)**2 > 1, 0, 1)
    non_propagating_mask = np.fft.ifftshift(non_propagating_mask)
    propagating_power = np.sum(np.abs(fft_operator.output_array * non_propagating_mask)**2)/S
    print("FFT power: %.4e, Propagating power: %.4e, Non-propagating power: %.4e" % (fft_power, propagating_power, fft_power - propagating_power))
    ifft_input_array[:,:] = fft_operator.output_array * P_1_nat #element wise product
    print("AFTER")

    fft_power = np.sum(np.abs(ifft_operator.input_array)**2)/S
    non_propagating_mask = np.where((wavelength * nu_parallel)**2 > 1, 0, 1)
    non_propagating_mask = np.fft.ifftshift(non_propagating_mask)
    propagating_power = np.sum(np.abs(ifft_operator.input_array * non_propagating_mask)**2)/S
    print("FFT power: %.4e, Propagating power: %.4e, Non-propagating power: %.4e" % (fft_power, propagating_power, fft_power - propagating_power))
    
    ifft_operator()
    intermediate_fields[0] = np.fft.fftshift(ifft_operator.output_array.copy())
    print("Field intensity after first phase mask and propagation: %.4e" % np.sum(np.abs(intermediate_fields[0])**2))

    fft_input_array[:,:] = np.fft.ifftshift((np.fft.fftshift(ifft_operator.output_array) * np.exp(1j * phase_mask[S:].reshape((Nx, Ny)))))
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_2_nat #element wise product
    ifft_operator()
    intermediate_fields[1] = np.fft.fftshift(ifft_operator.output_array.copy())
    
    end_time = time.time()
    #plot intermediate fields
    make_2Dplot_of(intermediate_fields[0], choose_quantity="amplitude", save_name="intermediate_field_1_amplitude", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
    make_2Dplot_of(intermediate_fields[0], choose_quantity="phase", save_name="intermediate_field_1_phase", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
    make_2Dplot_of(intermediate_fields[1], choose_quantity="amplitude", save_name="intermediate_field_2_amplitude", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
    make_2Dplot_of(intermediate_fields[1], choose_quantity="phase", save_name="intermediate_field_2_phase", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
    print("Propagation finished in {:.6f} seconds.".format(end_time - start_time))


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
    return np.concatenate((grad_C_phi_1, grad_C_phi_2))

def _compute_cost():

    forward_propagate()

    output_field = intermediate_fields[-1].flatten()

    #circular_mask = np.where(rho.flatten() < 20*6e-6, 1, 0) 
    C_s = np.abs(np.sum(output_field * np.conj(target_Efield))) ** 2
    C_s = np.real(np.sum(C_s))

    C_t = C_s

    return C_t   

opt_history = []

def cost_fun(x, grad):
    ''' Cost function to be minimized.
    Args:
        x: input normalized parameter field (flattened)
        grad: gradient of the cost function with respect to x, modified in place (flattened)
    Returns:
        The cost function value for the input parameter field x.
    '''
    w_norm[:] = x

    phase_mask[:] = phase_given_w(w_norm)

    C = _compute_cost() 

    opt_history.append(C)

    plt.figure()
    plt.plot(opt_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.title('Optimization History')
    plt.tight_layout()
    plt.savefig("results/modeconv1_optimization_history.pdf")
    plt.close()
    
    if grad.size > 0:
        grad[:] = adjoint_propagate() * 2 * np.pi
        #print the gradient magnitude
        grad_magnitude = np.abs(grad[S:]).reshape((Nx, Ny))
        make_2Dplot_of(grad_magnitude, choose_quantity="amplitude", save_name="gradient_magnitude", plot_zoom_x=all_view_x, plot_zoom_y=all_view_y)
    return C

if __name__ == "__main__":
    #initialize nlopt solver
    solver = nlopt.opt(nlopt.LD_CCSAQ, 2*S)
    
    solver.set_lower_bounds(mask_lower_bounds.flatten())
    solver.set_upper_bounds(mask_upper_bounds.flatten())
    solver.set_max_objective(cost_fun)
    solver.set_maxeval(opt_max_eval)
    solver.set_param("dual_ftol_rel", 1e-7)
    solver.set_param("verbosity",1)

    print("Starting optimization...")
    start = True
    if start:
        w_norm[:] = solver.optimize(w_norm)
    print("Optimization completed.")

    #verify the results
    phase_mask = phase_given_w(w_norm)
    forward_propagate()
    output_field = intermediate_fields[-1]
    
    input_power = np.sum(np.abs(input_field)**2)
    output_power = np.sum(np.abs(output_field)**2)

    print("Norm. input power: %.4e" % input_power)
    print("Norm. output power: %.4e" % output_power)
    #save the output field for later use
    np.savetxt("results/modeconv1_optimized_output_field.txt", output_field.reshape((Nx, Ny)))
    # Plot results in individual PDF figures
    
    # Reshape fields to 2D for visualization
    output_field_2d = output_field.reshape((Nx, Ny))
    phase_mask_1 = phase_given_w(w_norm[:S]).reshape((Nx, Ny))
    phase_mask_2 = phase_given_w(w_norm[S:]).reshape((Nx, Ny))
    
    a = 6e-6
    d = 10 * a * 1e6

    # Plot: Output field amplitude and phase
    make_2Dplot_of(output_field_2d, choose_quantity="amplitude", save_name="output_field_amplitude", plot_zoom_x=d, plot_zoom_y=d)
    make_2Dplot_of(output_field_2d, choose_quantity="phase", save_name="output_field_phase", plot_zoom_x=d, plot_zoom_y=d)
    
    phase_mask_zoom = 150
    # Plot: Phase mask 1
    make_2Dplot_of(phase_mask_1, choose_quantity="phase_mask", save_name="phase_mask_1", plot_zoom_x=phase_mask_zoom, plot_zoom_y=phase_mask_zoom)
    # Plot: Phase mask 2
    make_2Dplot_of(phase_mask_2, choose_quantity="phase_mask", save_name="phase_mask_2", plot_zoom_x=phase_mask_zoom, plot_zoom_y=phase_mask_zoom)
    
    # Plot: Cuts along y=0 axis - Amplitude comparison
    y_center_idx = Ny // 2
    output_amplitude_cut = np.abs(output_field_2d[y_center_idx, :]) / np.max(np.abs(output_field_2d[y_center_idx, :]))
    target_amplitude_cut = np.abs(target_Efield_2d[y_center_idx, :]) / np.max(np.abs(target_Efield_2d[y_center_idx, :]))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs*1e6, output_amplitude_cut, label='Output Field', linewidth=2)
    ax.plot(xs*1e6, target_amplitude_cut, label='Target Field', linewidth=2)
    ax.set_xlabel('x [um]')
    ax.set_xlim(-d,d)
    ax.set_ylabel('Amplitude')
    ax.set_title('Norm. Field Amplitude along y=0 axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/amplitude_cut_comparison.pdf")
    plt.close()
    
    # Plot 8: Cuts along y=0 axis - Phase comparison
    output_phase_cut = (np.angle(output_field_2d[y_center_idx, :]) + 2*np.pi) % (2*np.pi)
    target_phase_cut = (np.angle(target_Efield_2d[y_center_idx, :]) + 2*np.pi) % (2*np.pi)
    x_ax = xs*1e6
    make_1Dplot_of([x_ax, x_ax], [output_phase_cut, target_phase_cut], save_name="phase_cut_comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs*1e6, output_phase_cut, label='Output Field', linewidth=2)
    ax.plot(xs*1e6, target_phase_cut, label='Target Field', linewidth=2)
    ax.set_xlabel('x [um]')
    ax.set_ylabel('Phase (rad)')
    ax.set_xlim(-d,d)
    ax.set_title('Field Phase along y=0 axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/phase_cut_comparison.pdf")
    plt.close()

    # Plot 9: Intermediate field before second phase mask
    make_2Dplot_of(intermediate_fields[0], choose_quantity="amplitude", save_name="intermediate_field_amplitude", plot_zoom_x=d, plot_zoom_y=d)
    
    #save the phase masks for later use
    np.savetxt("results/optimized_phase_mask_1.txt", phase_mask_1)
    np.savetxt("results/optimized_phase_mask_2.txt", phase_mask_2)

    print("All plots saved to results/ folder.")

