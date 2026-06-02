from autograd import numpy as npa
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image
import nlopt

wavelength = 1
k0 = 2 * np.pi / wavelength

size_x = 100 * wavelength
size_y = 100 * wavelength
res_x = 16 #number of pixels per wavelength
res_y = res_x 

Nx = int(res_x * size_x) + 1
Ny = int(res_y * size_y) + 1
S = Nx * Ny

norm_w_min = 0
norm_w_max = 1
w_norm = np.ones((S)) * 0.5 #normalized geometrical parameters in [0,1]
w_min = 0.4
w_max = 1

xs = np.linspace(-size_x/2, size_x/2, Nx)
ys = np.linspace(-size_y/2, size_y/2, Ny)
sampling_period = xs[1] - xs[0]

z_prop = 300 #propagation distance

input_field = np.ones((S))

opt_max_eval = 50

def geom_transform(w_norm):
    '''Applies a linear mapping to the input parameter field w_norm,
    to obtain true geometrical parameters. 
    Args:
        w_norm: normalized geometrical parameters in [0,1] (flattened)
    Returns:
        The true geometrical parameters corresponding to the input normalized parameters (flattened)
    '''
    w = w_min + (w_max - w_min) * w_norm
    return w

def phase_given_w(w):
    '''Returns the phase shift given the geometrical parameters w.
    Args:
        w: geometrical parameters (flattened)
    Returns:
        The phase shift corresponding to the geometrical parameters w (flattened)'''
    #for simplicity, we assume that the phase shift is proportional to w
    return 2 * np.pi * (w - w_min) / (w_max - w_min)

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
    '''Returns the pattern intensity value at the specified (xs,ys) locations, with a scale factor.
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

target_intensity = intensity_desired_fn_pattern(xs, ys)
mean_target_intensity = np.mean(target_intensity)

def forward_propagate(phase_mask):
    '''Propagates the input field, after applying the phase mask

    Args:
        phase_mask: the phase mask to be applied to the input field (flattened)
    Returns:        
        A tuple with output_field (flattened) and corresponding propagation matrix P (2D)
    '''
    field_after_mask = input_field * np.exp(1j * phase_mask)
    field_after_mask_2d = field_after_mask.reshape((Nx, Ny))
    field_fft = np.fft.fftshift(np.fft.fft2(field_after_mask_2d))

    ks = 2 * np.pi / sampling_period
    kappas = np.linspace(-field_fft.shape[0]//2, field_fft.shape[0]//2-1, field_fft.shape[0]) * ks / field_fft.shape[0]
    KX, KY = np.meshgrid(kappas, kappas)
    K_parallel = np.sqrt(KX**2 + KY**2)
    nu_parallel = K_parallel / (2 * np.pi)
    phase_factor = k0 * z_prop * np.emath.sqrt(1 - (wavelength * nu_parallel) ** 2)

    P = np.exp(1j * phase_factor)
    fft_field_propagated = np.multiply(field_fft,P)

    output_field_2d = np.fft.ifft2(np.fft.ifftshift(fft_field_propagated))
    output_field = output_field_2d.flatten()

    return output_field, P

def adjoint_propagate(output_field, P, phase_mask):
    '''Backpropagates the output field using the adjoint of the propagation matrix P.
    Args:
        output_field: the output field to be backpropagated (flattened)
        P: the propagation matrix used in the forward propagation
    Returns:
        The backpropagated field (flattened)
    '''
    output_intensity = output_field * np.conj(output_field)
    adjoint_field = (output_intensity - target_intensity) * output_field
    adjoint_field_2d = adjoint_field.reshape((Nx, Ny))
    adjoint_field_fft = np.fft.fftshift(np.fft.fft2(adjoint_field_2d))
    P_dagger = P.T.conj()
    adjoint_field_propagated_fft = adjoint_field_fft * P_dagger
    adjoint_field_propagated_2d = np.fft.ifft2(np.fft.ifftshift(adjoint_field_propagated_fft))
    
    #apply the conjugate of the phase mask to backpropagate through it
    adjoint_field_propagated = adjoint_field_propagated_2d.flatten() * np.exp(-1j * phase_mask) 
    return adjoint_field_propagated

def cost_fun(x, grad):
    ''' Cost function to be minimized.
    Args:
        x: input normalized parameter field (flattened)
        grad: gradient of the cost function with respect to x, modified in place (flattened)
    Returns:
        The cost function value for the input parameter field x.
    '''
    w_norm[:] = x
    true_geom = geom_transform(w_norm)
    phase_mask = phase_given_w(true_geom)

    output_field, P = forward_propagate(phase_mask)
    
    output_intensity = np.real(output_field * np.conj(output_field))
    C_s = (output_intensity - target_intensity) ** 2 
    C = np.sum(C_s)

    #make a plot of three items: 1 the target intensity, 2 the output intensity, and 3 the cost function value as a function of x and y
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(target_intensity.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title('Target Intensity')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.subplot(1, 3, 2)
    plt.imshow(output_intensity.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title('Output Intensity')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.subplot(1, 3, 3)
    plt.imshow(C_s.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title('Cost Function Value')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.tight_layout()
    plt.savefig("results/cost_function_visualization.pdf")
    #plt.show()
    plt.close()        

    if grad.size > 0:
        adj_field_prop = adjoint_propagate(output_field, P, phase_mask)
        grad[:] = 4 * np.real(-1j * input_field.reshape((Nx,Ny)).T.conj().flatten() * adj_field_prop)
    
    return C

if __name__ == "__main__":
    #initialize nlopt solver
    solver = nlopt.opt(nlopt.LD_CCSAQ, S)
    solver.set_lower_bounds(norm_w_min)
    solver.set_upper_bounds(norm_w_max)
    solver.set_min_objective(cost_fun)
    solver.set_maxeval(opt_max_eval)
    solver.set_param("dual_ftol_rel", 1e-7)
    solver.set_param("verbosity",1)

    print("Starting optimization...")
    w_norm[:] = solver.optimize(w_norm)
    print("Optimization completed.")

    #verify the results
    true_geom = geom_transform(w_norm)
    phase_mask = phase_given_w(true_geom)
    output_field, _ = forward_propagate(phase_mask)
    output_intensity = np.real(output_field * np.conj(output_field))
    #make a plot of the phase mask, with phase wrap between -pi and pi
    plt.figure()
    plt.imshow(phase_mask.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower') 
    plt.colorbar()
    plt.title('Phase Mask')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.savefig("results/optimized_phase_mask.pdf")
    plt.show()
    plt.close()
    #make a subplot to compare the output intensity with the target intensity
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(output_intensity.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title('Output Intensity')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.subplot(1, 2, 2)
    plt.imshow(target_intensity.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.title('Target Intensity')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.tight_layout()
    plt.savefig("results/comparison_output_target.pdf")
    plt.show()