#from jax import numpy as np, grad
#from jax.numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import List, Tuple, Union
import time

import pyfftw
#pyFFTW setup
pyfftw.config.NUM_THREADS = 4
pyfftw.interfaces.cache.enable()
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'

import nlopt

import numpy as np
import matplotlib.pyplot as plt

from scipy.special import jv, kv
ArrayLikeType = Union[List, Tuple, np.ndarray]

wavelength = 1.55e-6
k0 = 2 * np.pi / wavelength

opt_max_eval = 250

size_x = 1024 * wavelength
size_y = 1024 * wavelength
res_x = 3 / 1e-6 #number of pixels per unit-length
res_y = res_x
ds = 1 / res_x

Nx = int(round(res_x * size_x)) + 1
Ny = int(round(res_y * size_y)) + 1 
S = Nx * Ny

######################
#plan pyfftw objects for forward and inverse FFTs
fft_input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
fft_output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
ifft_input_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)
ifft_output_array = pyfftw.empty_aligned((Nx, Ny), dtype='complex128', n=16)

fft_operator = pyfftw.FFTW(fft_input_array, fft_output_array, axes=(0,1), direction='FFTW_FORWARD')
ifft_operator = pyfftw.FFTW(ifft_input_array, ifft_output_array, axes=(0,1), direction='FFTW_BACKWARD')
######################

norm_phase_min = 0
norm_phase_max = 1

phase_min = 0
phase_max = 2 * np.pi

xs = np.linspace(-size_x/2, size_x/2, Nx)
ys = np.linspace(-size_y/2, size_y/2, Ny)

X, Y = np.meshgrid(xs, ys)
rho = np.sqrt(X**2 + Y**2)

sampling_period = xs[1] - xs[0]

d1 = 4000e-6 #propagation distance
d2 = 1000e-6 #propagation distance
d = [d1, d2] #d = [d1,d2] d1:distance MS1-MS2, d2: distance MS2-target

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
    '''Returns the flattened target fiber mode pattern'''
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
            phi = np.arctan2(Y[i,j], X[i,j])
            if r < a:#inside core
                E_field[i,j] = jv(n_mode_sel,chi_co * r) / jv(n_mode_sel, chi_co * a) * np.cos(n_mode_sel * phi)
            else:#outside core
                E_field[i,j] = kv(n_mode_sel, chi_cl * r) / kv(n_mode_sel, chi_cl * a) * np.cos(n_mode_sel * phi)

    return E_field.flatten()

###########################
#initialize input field
beam_waist = 70e-6
gaussian_field = np.exp(-rho**2 / (beam_waist)**2) #Gaussian input field, flattened
input_field = (gaussian_field / np.sqrt(np.sum(np.abs(gaussian_field)**2))).flatten() #normalize the input field to have power = 1
###########################

###########################
#initialize target field
target_Efield = get_fiber_mode_pattern() / np.sqrt(np.sum(np.abs(get_fiber_mode_pattern())**2)) #normalized target field (intensity = 1)
###########################

###########################
#initialize the normalized phase masks and global phase mask
w_norm = np.zeros((2*S)) #concatenated normalized parameters for both masks
X, Y = np.meshgrid(np.linspace(-size_x/2, size_x/2, Nx), np.linspace(-size_y/2, size_y/2, Ny))
rho = np.sqrt(X**2 + Y**2)
circular_mask_1 = np.where(rho.flatten() < 10*6e-6, 0.5, 0) 
w_norm[:S] += circular_mask_1
target_pattern_phase = (np.angle(target_Efield) + 2 * np.pi) % (2 * np.pi) #make sure the phase is between 0 and 2pi
w_norm[S:] += target_pattern_phase / (2 * np.pi) #normalize the target phase to be between 0 and 1
phase_mask = phase_given_w(w_norm)
phase_mask_1 = phase_mask[:S]
phase_mask_2 = phase_mask[S:]
###########################

###########################
#setting up spatial frequencies
ks = 2 * np.pi / sampling_period
kappas = np.arange(-Nx//2, Nx//2) * ks / Nx
KX, KY = np.meshgrid(kappas, kappas)
K_parallel = np.sqrt(KX**2 + KY**2)
nu_parallel = K_parallel / (2 * np.pi)

phase_factor = k0 * d1 * np.sqrt(1 - (wavelength * nu_parallel) ** 2 + 0*1j)
P_1 = np.exp(1j * phase_factor)
P_1_nat = np.fft.ifftshift(P_1)
phase_factor_2 = k0 * d2 * np.sqrt(1 - (wavelength * nu_parallel) ** 2 + 0*1j)
P_2 = np.exp(1j * phase_factor_2)
P_2_nat = np.fft.ifftshift(P_2)
###########################

intermediate_fields = [None, None]
def forward_propagate() -> None:
    '''Propagates the input field through the system with the given phase mask.
    Args:
        phase_mask: the concatenated phase mask of the system. Masks are flattened and in order. Any mask has always S elements.
    Returns:
        The output field after propagation (flattened) and the propagation matrices P used in the forward propagation.
    '''
    print("Starting propagating...")
    start_time = time.time()

    fft_input_array[:,:] = (input_field * np.exp(1j * phase_mask_1)).reshape((Nx, Ny))
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_1_nat #element wise product
    ifft_operator()
    intermediate_fields[0] = ifft_operator.output_array.copy()

    fft_input_array[:,:] = (ifft_operator.output_array.flatten() * np.exp(1j * phase_mask_2)).reshape((Nx, Ny))
    fft_operator()
    ifft_input_array[:,:] = fft_operator.output_array * P_2_nat #element wise product
    ifft_operator()
    intermediate_fields[1] = ifft_operator.output_array.copy()
    
    end_time = time.time()
    print("Propagation finished in {:.6f} seconds.".format(end_time - start_time))


def adjoint_propagate():
    '''Backpropagates the output field using the adjoint of the propagation matrix P.
    Args:
        output_field: the output field to be backpropagated (flattened)
        P: the propagation matrix used in the forward propagation
        phase_mask: the phase mask used in the forward propagation (flattened)
    Returns:
        The backpropagated field (flattened)
    '''
    print("Computing adjoint...")
    start_time = time.time()
    phase_mask_1 = phase_mask[:S]
    phase_mask_2 = phase_mask[S:]
    
    output_field_1 = intermediate_fields[0]
    input_field_2 = output_field_1
    output_field_2 = intermediate_fields[1]
    
    P_1_dagger = P_1.T.conj()
    P_2_dagger = P_2.T.conj()
    
    Phi_1 = np.exp(1j * phase_mask_1)
    Phi_2 = np.exp(1j * phase_mask_2)
    Phi_1_dagger = np.exp(-1j * phase_mask_1)
    Phi_2_dagger = np.exp(-1j * phase_mask_2)
    
    #compute the adjoint source
    adjoint_field = np.sum(output_field_2 * np.conj(target_Efield)) * target_Efield
    adjoint_field_2d = adjoint_field.reshape((Nx, Ny))
    
    adjoint_field_fft = np.fft.fftshift(np.fft.fft2(adjoint_field_2d))
    pd2 = adjoint_field_fft * P_2_dagger #element wise
    adjoint_field_2_propagated_2d = np.fft.ifft2(np.fft.ifftshift(pd2))
    adjoint_field_2_propagated = adjoint_field_2_propagated_2d.flatten() * Phi_2_dagger
    adjoint_field_2_propagated_2d = adjoint_field_2_propagated.reshape((Nx, Ny))
    adjoint_field_2_propagated_fft = np.fft.fftshift(np.fft.fft2(adjoint_field_2_propagated_2d))
    pd1 = adjoint_field_2_propagated_fft * P_1_dagger #element wise
    adjoint_field_1_propagated_2d = np.fft.ifft2(np.fft.ifftshift(pd1))
    adjoint_field_1_propagated = adjoint_field_1_propagated_2d.flatten() * Phi_1_dagger

    grad_C_phi_1 = 2 * np.real(-1j * input_field.reshape((Nx,Ny)).T.conj().flatten() * adjoint_field_1_propagated)

    grad_C_phi_2 = 2 * np.real(-1j * output_field_1.reshape((Nx,Ny)).T.conj().flatten() * adjoint_field_2_propagated)    

    grads = np.concatenate((grad_C_phi_1, grad_C_phi_2))
    
    end_time = time.time()
    print("Adjoint finished in {:.6f} seconds.".format(end_time - start_time))
    return grads

def _compute_cost():

    forward_propagate()

    output_field = intermediate_fields[-1]

    circular_mask = np.where(rho.flatten() < 10*6e-6, 1, 0) 
    C_s = np.abs(np.sum(output_field*circular_mask * np.conj(target_Efield))) ** 2
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
    return C

if __name__ == "__main__":
    #initialize nlopt solver
    solver = nlopt.opt(nlopt.LD_CCSAQ, 2*S)
    solver.set_lower_bounds(norm_phase_min)
    solver.set_upper_bounds(norm_phase_max)
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
    target_field_2d = target_Efield.reshape((Nx, Ny))
    phase_mask_1 = phase_given_w(w_norm[:S]).reshape((Nx, Ny))
    phase_mask_2 = phase_given_w(w_norm[S:]).reshape((Nx, Ny))
    
    a = 6e-6
    d = 10 * a * 1e6
    # Plot 1: Output field amplitude
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.abs(output_field_2d),extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2), origin='lower', cmap='viridis')
    ax.set_title('Output Field Amplitude')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-d, d)
    ax.set_ylim(-d, d)
    ax.set_aspect('equal') 
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("results/output_field_amplitude.pdf")
    plt.close()
    
    # Plot 2: Output field phase
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow((np.angle(output_field_2d)+2*np.pi)%(2*np.pi),extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2), origin='lower', cmap='hsv')
    ax.set_title('Output Field Phase')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-d, d)
    ax.set_ylim(-d, d)
    ax.set_aspect('equal') 
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("results/output_field_phase.pdf")
    plt.close()
    
    # Plot 3: Target field amplitude
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.abs(target_field_2d),extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2), origin='lower', cmap='viridis')
    ax.set_title('Target Field Amplitude')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-d, d)
    ax.set_ylim(-d, d)
    ax.set_aspect('equal') 
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("results/target_field_amplitude.pdf")
    plt.close()
    
    # Plot 4: Target field phase
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow((np.angle(target_field_2d)+2*np.pi)%(2*np.pi),extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2), origin='lower', cmap='hsv')
    ax.set_title('Target Field Phase')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-d, d)
    ax.set_ylim(-d, d)
    ax.set_aspect('equal') 
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("results/target_field_phase.pdf")
    plt.close()
    
    # Plot 5: Phase mask 1
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow((phase_mask_1+2*np.pi)%(2*np.pi),extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2), origin='lower', cmap='viridis')
    ax.set_title('Phase Mask 1')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal') 
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("results/phase_mask_1.pdf")
    plt.close()
    
    # Plot 6: Phase mask 2
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow((phase_mask_2+2*np.pi)%(2*np.pi),extent=(-size_x*1e6/2, size_x*1e6/2, -size_y*1e6/2, size_y*1e6/2), origin='lower', cmap='viridis')
    ax.set_title('Phase Mask 2')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal') 
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("results/phase_mask_2.pdf")
    plt.close()
    
    # Plot 7: Cuts along y=0 axis - Amplitude comparison
    y_center_idx = Ny // 2
    output_amplitude_cut = np.abs(output_field_2d[y_center_idx, :])
    target_amplitude_cut = np.abs(target_field_2d[y_center_idx, :])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs*1e6, output_amplitude_cut, label='Output Field', linewidth=2)
    ax.plot(xs*1e6, target_amplitude_cut, label='Target Field', linewidth=2)
    ax.set_xlabel('x (m)')
    ax.set_xlim(-d,d)
    ax.set_ylabel('Amplitude')
    ax.set_title('Field Amplitude along y=0 axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/amplitude_cut_comparison.pdf")
    plt.close()
    
    # Plot 8: Cuts along y=0 axis - Phase comparison
    output_phase_cut = (np.angle(output_field_2d[y_center_idx, :]) + 2*np.pi) % (2*np.pi)
    target_phase_cut = (np.angle(target_field_2d[y_center_idx, :]) + 2*np.pi) % (2*np.pi)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs*1e6, output_phase_cut, label='Output Field', linewidth=2)
    ax.plot(xs*1e6, target_phase_cut, label='Target Field', linewidth=2)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Phase (rad)')
    ax.set_xlim(-d,d)
    ax.set_title('Field Phase along y=0 axis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/phase_cut_comparison.pdf")
    plt.close()
    
    #save the phase masks for later use
    np.savetxt("results/optimized_phase_mask_1.txt", phase_mask_1)
    np.savetxt("results/optimized_phase_mask_2.txt", phase_mask_2)

    print("All plots saved to results/ folder.")

