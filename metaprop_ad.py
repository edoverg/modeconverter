#from jax import numpy as npa, grad
#from jax.numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import List, Tuple, Union
from autograd import numpy as npa, tensor_jacobian_product, grad as gd
import meep.adjoint as mpa
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from PIL import Image
import nlopt
ArrayLikeType = Union[List, Tuple, np.ndarray]

wavelength = 1.55
k0 = 2 * np.pi / wavelength

size_x = 300 * wavelength
size_y = 300 * wavelength
res_x = 3 #number of pixels per unit-length
res_y = res_x 

Nx = int(res_x * size_x) + 1
Ny = int(res_y * size_y) + 1
S = Nx * Ny

norm_phase_min = 0
norm_phase_max = 1
w_norm = np.ones((S)) * 0.5 #normalized phase parameters in [0,1]
phase_min = 0
phase_max = 2 * np.pi

xs = np.linspace(-size_x/2, size_x/2, Nx)
ys = np.linspace(-size_y/2, size_y/2, Ny)
sampling_period = xs[1] - xs[0]

z_prop = 300 #propagation distance

input_field = np.ones((S))

opt_max_eval = 200

def phase_given_w(w):
    '''Returns the phase shift given the design parameters w.
    Args:
        w: design parameters (flattened)
    Returns:
        The phase shift corresponding to the geometrical parameters w (flattened)'''
    #for simplicity, we linearly map w (0,1) to a phase shift (0, 2pi)
    return 2 * npa.pi * (w)

def get_pattern() -> np.ndarray:
    '''Imports the intensity pattern from file'''
    pattern_name = 'unipd.png'
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
    pattern_values = np.rot90(np.rot90(np.rot90(pattern_values)))

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
    plt.savefig("results/test1_target_intensity_pattern.pdf")
    plt.close()
    return pattern_to_return

target_intensity = intensity_desired_fn_pattern(xs, ys)
mean_target_intensity = np.mean(target_intensity)


filter_radius_um = 0.5
def _centered(arr: np.ndarray, newshape: ArrayLikeType) -> np.ndarray:
    """Formats the output of an FFT to center the zero-frequency component.

    A helper function borrowed from SciPy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270

    Args:
        arr: output array from an FFT operation.
        newshape: 1d array with two elements (integers) specifying the dimensions
            of the array to be returned.

    Returns:
        The input array with the zero-frequency component as the central element.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]

def _edge_pad(arr: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """Border-pads the edges of an array.

    Used to preprocess the design weights prior to convolution with the filter.
    Border padding an image will set the value of each padded pixel equal to
    the value of the nearest pixel in the image. Used to implement feature-
    preserving convolution filters that prevent unwanted edge effects.

    Args:
        arr: 2d array whose borders contain the values to use for padding
        pad: 2x2 array of integers indicating the size
            of the borders to pad the array with.

    Returns:
        A 2d array with border padding.
    """
    # fill sides
    left = npa.tile(arr[0, :], (pad[0][0], 1))
    right = npa.tile(arr[-1, :], (pad[0][1], 1))
    top = npa.tile(arr[:, 0], (pad[1][0], 1)).transpose()
    bottom = npa.tile(arr[:, -1], (pad[1][1], 1)).transpose()

    # fill corners
    top_left = npa.tile(arr[0, 0], (pad[0][0], pad[1][0]))
    top_right = npa.tile(arr[-1, 0], (pad[0][1], pad[1][0]))
    bottom_left = npa.tile(arr[0, -1], (pad[0][0], pad[1][1]))
    bottom_right = npa.tile(arr[-1, -1], (pad[0][1], pad[1][1]))

    if pad[0][0] > 0 and pad[0][1] > 0 and pad[1][0] > 0 and pad[1][1] > 0:
        return npa.concatenate(
            (
                npa.concatenate((top_left, top, top_right)),
                npa.concatenate((left, arr, right)),
                npa.concatenate((bottom_left, bottom, bottom_right)),
            ),
            axis=1,
        )
    elif pad[0][0] == 0 and pad[0][1] == 0 and pad[1][0] > 0 and pad[1][1] > 0:
        return npa.concatenate((top, arr, bottom), axis=1)
    elif pad[0][0] > 0 and pad[0][1] > 0 and pad[1][0] == 0 and pad[1][1] == 0:
        return npa.concatenate((left, arr, right), axis=0)
    elif pad[0][0] == 0 and pad[0][1] == 0 and pad[1][0] == 0 and pad[1][1] == 0:
        return arr
    else:
        raise ValueError("At least one of the padding numbers is invalid.")

def _quarter_to_full_kernel(arr: np.ndarray, pad_to: np.ndarray) -> np.ndarray:
    """Constructs the full kernel from its nonnegative quadrant.

    Args:
        arr: 2d input array representing the nonnegative quadrant of a
            filter kernel with C4v symmetry.
        pad_to: 1d array with two elements (integers) specifying the size
            of the zero padding.

    Returns:
        The complete kernel.
    """
    pad_size = pad_to - 2 * np.array(arr.shape) + 1

    top = np.zeros((pad_size[0], arr.shape[1]))
    bottom = np.zeros((pad_size[0], arr.shape[1] - 1))
    middle = np.zeros((pad_to[0], pad_size[1]))

    top_left = arr[:, :]
    top_right = npa.flipud(arr[1:, :])
    bottom_left = npa.fliplr(arr[:, 1:])
    bottom_right = npa.flipud(
        npa.fliplr(arr[1:, 1:])
    )  # equivalent to flip, but flip is incompatible with autograd

    return npa.concatenate(
        (
            npa.concatenate((top_left, top, top_right)),
            middle,
            npa.concatenate((bottom_left, bottom, bottom_right)),
        ),
        axis=1,
    )

def conic_filter(phase_mask,radius,size_x,size_y,resolution):
    '''Applies a conic filter to the input phase mask. The filter is defined as a 
    conic region of a specified radius, where the values are averaged. 

    Args:
        phase_mask: the input phase mask to be filtered (2D)
        radius: the radius of the conic filter in micrometers
        size_x: the size of the phase mask in x direction in micrometers
        size_y: the size of the phase mask in y direction in micrometers
        resolution: the resolution of the phase mask in pixels per micrometer
    Returns:
        The filtered phase mask (2D)
    '''
    Nx = int(round(size_x * resolution)) + 1
    Ny = int(round(size_y * resolution)) + 1
    
    xv = np.arange(0, size_x / 2, 1 / resolution) if resolution > 0 else [0]
    yv = np.arange(0, size_y / 2, 1 / resolution) if resolution > 0 else [0]

    X, Y = npa.meshgrid(xv, yv)
    
    h = npa.where(X**2 + Y**2 < radius**2, (1 - np.sqrt(abs(X**2 + Y**2)) / radius),0)
    #h = npa.where(X**2 + Y**2 < radius**2, 1, 0)
    h = _quarter_to_full_kernel(h, 3 * np.array([Nx, Ny]))
    h = h / npa.sum(h)
    H = npa.fft.fft2(h)

    phase_mask_2d = phase_mask.reshape(Nx,Ny)
    phase_mask_padded = _edge_pad(phase_mask_2d, ((Nx, Nx), (Ny, Ny)))
    phase_mask_fft = npa.fft.fft2(phase_mask_padded)
    
    Y = phase_mask_fft * H
    
    y = _centered(
        npa.real(npa.fft.ifft2(Y)),
        (Nx, Ny)
    )

    return y.flatten()

def phase_filter(phase_mask):
    '''Applies a conic filter on the phase mask'''
    phase_mask_filtered = conic_filter(
        phase_mask.reshape(Nx,Ny), #row major mapping 
        filter_radius_um,
        size_x,
        size_y,
        res_x,
    )
    return phase_mask_filtered.flatten()

def forward_propagate(phase_mask):
    '''Propagates the input field, after applying the phase mask

    Args:
        phase_mask: the phase mask to be applied to the input field (flattened)
    Returns:        
        A tuple with output_field (flattened) and corresponding propagation matrix P (2D)
    '''
    field_after_mask = input_field * npa.exp(1j * phase_mask)
    field_after_mask_2d = field_after_mask.reshape((Nx, Ny))
    field_fft = npa.fft.fftshift(npa.fft.fft2(field_after_mask_2d))

    ks = 2 * npa.pi / sampling_period
    kappas = npa.arange(-field_fft.shape[0]//2, field_fft.shape[0]//2) * ks / field_fft.shape[0]
    KX, KY = npa.meshgrid(kappas, kappas)
    K_parallel = npa.sqrt(KX**2 + KY**2)
    nu_parallel = K_parallel / (2 * npa.pi)
    phase_factor = k0 * z_prop * npa.sqrt(1 - (wavelength * nu_parallel) ** 2 + 0*1j)

    P = npa.exp(1j * phase_factor)
    fft_field_propagated = npa.multiply(field_fft,P) #element wise product

    output_field_2d = npa.fft.ifft2(npa.fft.ifftshift(fft_field_propagated))
    output_field = output_field_2d.flatten()

    return output_field, P

def adjoint_propagate(x):
    '''Backpropagates the output field using the adjoint of the propagation matrix P.
    Args:
        output_field: the output field to be backpropagated (flattened)
        P: the propagation matrix used in the forward propagation
        phase_mask: the phase mask used in the forward propagation (flattened)
    Returns:
        The backpropagated field (flattened)
    '''
    phase_mask = phase_given_w(x)
    field_after_mask = input_field * npa.exp(1j * phase_mask)
    field_after_mask_2d = field_after_mask.reshape((Nx, Ny))
    field_fft = npa.fft.fftshift(npa.fft.fft2(field_after_mask_2d))

    ks = 2 * npa.pi / sampling_period
    kappas = npa.arange(-field_fft.shape[0]//2, field_fft.shape[0]//2) * ks / field_fft.shape[0]
    KX, KY = npa.meshgrid(kappas, kappas)
    K_parallel = npa.sqrt(KX**2 + KY**2)
    nu_parallel = K_parallel / (2 * npa.pi)
    phase_factor = k0 * z_prop * npa.sqrt(1 - (wavelength * nu_parallel) ** 2 + 0*1j)

    P = npa.exp(1j * phase_factor)
    fft_field_propagated = npa.multiply(field_fft,P) #element wise product

    output_field_2d = npa.fft.ifft2(npa.fft.ifftshift(fft_field_propagated))
    output_field = output_field_2d.flatten()

    output_intensity = output_field * np.conj(output_field)
    adjoint_field = (output_intensity - target_intensity) * output_field
    adjoint_field_2d = adjoint_field.reshape((Nx, Ny))
    adjoint_field_fft = npa.fft.fftshift(npa.fft.fft2(adjoint_field_2d))
    P_dagger = P.T.conj()
    adjoint_field_propagated_fft = adjoint_field_fft * P_dagger #element wise
    adjoint_field_propagated_2d = npa.fft.ifft2(npa.fft.ifftshift(adjoint_field_propagated_fft))
    
    #apply the conjugate of the phase mask to backpropagate through it
    adjoint_field_propagated = adjoint_field_propagated_2d.flatten() * np.exp(-1j * phase_mask) 
    return adjoint_field_propagated

phase_mask_filtered = np.zeros((S))
def _compute_cost(phase_mask):

    output_field, P = forward_propagate(phase_mask)
    
    output_intensity = npa.real(output_field * npa.conj(output_field))
    C_s = (output_intensity - target_intensity) ** 2

    C = npa.sum(C_s)

    return C

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

    phase_mask = phase_given_w(w_norm)

    phase_mask_filtered = phase_filter(phase_mask)

    cost_function_denominator = np.sum(np.ones((S)))

    C = _compute_cost(phase_mask_filtered) / cost_function_denominator
    opt_history.append(C)
    #make a plot of the optimization history
    plt.figure()
    plt.plot(opt_history)
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.title('Optimization History')
    plt.savefig("results/test1_optimization_history.pdf")
    plt.close()
    #first round of backpropagation
    grad_test = gd(_compute_cost)(phase_mask_filtered)

    #second round of backpropagation
    full_grad = tensor_jacobian_product(phase_filter)(phase_mask,grad_test)
    #print("grad from autograd",grad_test)
    
    if grad.size > 0:
        #adj_field_prop = adjoint_propagate(x)
        #grad[:] = 4 * np.real(-1j * input_field.reshape((Nx,Ny)).T.conj().flatten() * adj_field_prop)
        grad[:] = full_grad * 2 * np.pi / cost_function_denominator
    #print("grad from theo",grad)
    #print("grad diff",grad - grad_test)
    return C

if __name__ == "__main__":
    #initialize nlopt solver
    solver = nlopt.opt(nlopt.LD_CCSAQ, S)
    solver.set_lower_bounds(norm_phase_min)
    solver.set_upper_bounds(norm_phase_max)
    solver.set_min_objective(cost_fun)
    solver.set_maxeval(opt_max_eval)
    solver.set_param("dual_ftol_rel", 1e-7)
    solver.set_param("verbosity",1)

    print("Starting optimization...")
    w_norm[:] = solver.optimize(w_norm)
    print("Optimization completed.")

    #verify the results
    phase_mask = phase_given_w(w_norm)
    filtered_mask = phase_filter(phase_mask)
    #save the phase_mask as raw 2D text
    np.savetxt("results/optimized_phase_mask.txt", filtered_mask.reshape((Nx, Ny)))
    
    output_field, _ = forward_propagate(filtered_mask)
    output_intensity = np.real(output_field * np.conj(output_field))
    #make a plot of the phase mask, with phase wrap between -pi and pi
    plt.figure()
    plt.imshow(phase_mask.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower') 
    plt.colorbar()
    plt.title('Phase Mask')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.savefig("results/test1_optimized_phase_mask.pdf")
    plt.show()
    plt.close()
    #make a subplot to compare the output intensity with the target intensity
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(output_intensity.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.clim(0,1)
    plt.title('Output Intensity')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.subplot(1, 2, 2)
    plt.imshow(target_intensity.reshape((Nx, Ny)), extent=(-size_x/2, size_x/2, -size_y/2, size_y/2), origin='lower', cmap='inferno')
    plt.colorbar()
    plt.clim(0,1)
    plt.title('Target Intensity')
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.tight_layout()
    plt.savefig("results/test1_comparison_output_target.pdf")
    #plt.show()
    plt.close()