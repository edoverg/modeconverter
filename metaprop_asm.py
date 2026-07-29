import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d import web

wavelength = 1.55e-6
k0 = 2 * np.pi / wavelength

unit_cell_pitch = 400e-9

Nx = 1024*2
Ny = Nx
S = Nx * Ny

size_x = unit_cell_pitch * Nx
size_y = unit_cell_pitch * Ny

norm_phase_min = 0
norm_phase_max = 1
w_norm = np.ones((S)) * 0.5 #normalized phase parameters in [0,1]
phase_min = 0
phase_max = 2 * np.pi

xs = np.linspace(-size_x/2, size_x/2, Nx)
ys = np.linspace(-size_y/2, size_y/2, Ny)

X, Y = np.meshgrid(xs, ys)
rho = np.sqrt(X**2 + Y**2)

sampling_period = xs[1] - xs[0]

#z_prop = 2.8e-6 #345e-6 #propagation distance
z_prop = 10e-6

beam_waist = 2e-6
gaussian_field = np.exp(-rho**2 / (beam_waist)**2) #Gaussian input field, flattened
input_field = gaussian_field.flatten() #not normalized input field

def phase_given_w(w):
    '''Returns the phase shift given the design parameters w.
    Args:
        w: design parameters (flattened)
    Returns:
        The phase shift corresponding to the geometrical parameters w (flattened)'''
    #for simplicity, we linearly map w (0,1) to a phase shift (0, 2pi)
    return 2 * np.pi * (w)

def forward_propagate(phase_mask):
    '''Propagates the input field, after applying the phase mask

    Args:
        phase_mask: the phase mask to be applied to the input field (flattened)
    Returns:        
        A tuple with output_field (flattened) and corresponding propagation matrix P (2D)
    '''
    #make sure the phase mask is flattened
    phase_mask = phase_mask.flatten()
    field_after_mask = input_field * np.exp(1j * phase_mask)
    field_after_mask_2d = field_after_mask.reshape((Nx, Ny))
    field_fft = np.fft.fftshift(np.fft.fft2(field_after_mask_2d))

    ks = 2 * np.pi / sampling_period
    kappas = np.arange(-field_fft.shape[0]//2, field_fft.shape[0]//2) * ks / field_fft.shape[0]
    KX, KY = np.meshgrid(kappas, kappas)
    K_parallel = np.sqrt(KX**2 + KY**2)
    nu_parallel = K_parallel / (2 * np.pi)
    phase_factor = k0 * z_prop * np.sqrt(1 - (wavelength * nu_parallel) ** 2 + 0*1j)
    #phase_factor = k0 * z_prop * (1 - 0.5*((wavelength * nu_parallel) ** 2 + 0*1j))
    P = np.exp(1j * phase_factor)
    fft_field_propagated = np.multiply(field_fft,P) #element wise product

    output_field_2d = np.fft.ifft2(np.fft.ifftshift(fft_field_propagated))
    output_field = output_field_2d.flatten()
   
    return output_field, P

if __name__ == "__main__":
    phase_mask_test = np.zeros((S)) #test with a zero phase mask
    #load the phase mask from file

    output_field, P = forward_propagate(phase_mask_test)

    plt.figure(figsize=(10, 10))
    plt.subplot(4, 2, 1)
    #plt.imshow(np.abs(field_fft), extent=(-ks/2, ks/2, -ks/2, ks/2))
    plt.imshow(np.abs(output_field.reshape((Nx,Ny))), extent=(-size_x/2*1e6, size_x/2*1e6, -size_y/2*1e6, size_y/2*1e6))
    plt.title("output")
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    
    plt.subplot(4, 2, 2)
    plt.imshow(np.abs(input_field.reshape((Nx, Ny)))**2, extent=(-size_x/2*1e6, size_x/2*1e6, -size_y/2*1e6, size_y/2*1e6))
    plt.title("input")
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    #add a slice of the output field along the x-axis and compare to the theoretical gaussian beam profile after propagation
    rayleigh_range = np.pi * beam_waist**2 / wavelength
    z = z_prop
    w_z = beam_waist * np.sqrt(1 + (z / rayleigh_range)**2)
    gaussian_profile = (beam_waist / w_z) * np.exp(- xs**2 / w_z**2)
    radius_of_curvature = z * (1 + (rayleigh_range / z)**2)
    gaussian_phase = (k0 * z + k0 * xs**2 / (2 * radius_of_curvature) - np.arctan(z / rayleigh_range)) % (2*np.pi)

    plt.subplot(4, 2, 3)
    plt.plot(xs*1e6, np.abs(output_field.reshape((Nx,Ny))[Nx//2, :])**2, 'b-', label='output')
    plt.plot(xs*1e6, gaussian_profile**2, 'r--', label='theory')
    plt.xlabel('x (um)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(loc='center left')
    plt.subplot(4, 2, 4)
    plt.plot(xs*1e6, np.abs(input_field.reshape((Nx,Ny))[Nx//2, :])**2, 'g--', label='input')
    plt.xlabel('x (um)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(loc='center left')

    plt.subplot(4,1,3)
    plt.plot(xs*1e6, np.angle(output_field.reshape((Nx,Ny))[Nx//2, :])%(2*np.pi), 'b-', label='output phase')
    plt.plot(xs*1e6, gaussian_phase, 'r--', label='theory phase')
    plt.plot(xs*1e6, np.angle(input_field.reshape((Nx,Ny))[Nx//2, :]), 'g--', label='input phase')
    plt.xlabel('x (um)')
    plt.ylabel('Phase (rad)')
    plt.legend(loc='center left')


    plt.subplot(4,1,4)
    plt.plot(xs*1e6, np.angle(output_field.reshape((Nx,Ny))[Nx//2, :])%(2*np.pi), 'b-', label='output phase')
    plt.plot(xs*1e6, gaussian_phase, 'r--', label='theory phase')
    plt.plot(xs*1e6, np.angle(input_field.reshape((Nx,Ny))[Nx//2, :]), 'g--', label='input phase')
    plt.xlabel('x (um)')
    plt.xlim(np.max(size_x/4*1e6), np.max(size_x/2*1e6))
    plt.ylabel('Phase (rad)')
    plt.legend(loc='center left')


    plt.tight_layout()
    #plt.savefig("propagation_verification.pdf")

    #load FDTD data from file
    lambda0 = 1.55
    freq0 = td.C_0 / lambda0 
    # Load the job metadata from file.
    job_loaded = web.Job.from_file("tidy3d/data/job.json")
    sim_data = job_loaded.load(path="tidy3d/data/testing_gauss_prop.hdf5")

    
    nf_field = sim_data["monitor_nf"].Ex.sel(f=freq0, z=2, method="nearest")
    src_field = sim_data["monitor_src"].Ex.sel(f=freq0, z=-1+1e-9, method="nearest")

    f = 10
    src_monitor_position = -0.8
    prj_field = sim_data["monitor_projection"].fields_cartesian.Ex.sel(f=freq0, z=f+src_monitor_position, method="nearest")

    fig, ax = plt.subplots(1,2)
    
    x_nf = nf_field.coords['x'].values
    x_src = src_field.coords['x'].values
    x_prj = prj_field.coords['x'].values
    N_nf_field = nf_field.shape[0]
    N_prj_field = prj_field.shape[0]

    nf_field_phase = ((np.angle(nf_field) + 2 * np.pi) % (2 * np.pi))[N_nf_field//2,:]
    nf_field_mag = (np.abs(nf_field) / np.max(np.abs(nf_field)))[N_nf_field//2,:]
    src_field_phase = ((np.angle(src_field) + 2 * np.pi) % (2 * np.pi))[N_nf_field//2,:]
    src_field_mag = (np.abs(src_field) / np.max(np.abs(src_field)))[N_nf_field//2,:]

    prj_field_phase = ((np.angle(prj_field) + 2 * np.pi) % (2 * np.pi))[N_prj_field//2,:]
    prj_field_mag = (np.abs(prj_field) / np.max(np.abs(prj_field)))[N_prj_field//2,:]

    output_field_2d = output_field.reshape((Nx, Ny))
    output_field_mag = (np.abs(output_field_2d) / np.max(np.abs(output_field_2d)))[Nx//2,:]
    output_field_phase = ((np.angle(output_field_2d) + 2 * np.pi) % (2 * np.pi))[Nx//2,:]

    #ax[0].plot(x_nf,nf_field_mag, '.b', label="FDTD magnitude")
    ax[0].plot(xs*1e6,output_field_mag, 'r', label="ASM magnitude")
    ax[0].plot(xs*1e6,gaussian_profile/np.max(gaussian_profile), '-.g', label="theory magnitude")
    ax[0].plot(x_prj,prj_field_mag, label="FDTD FF")
    ax[0].set_xlim(-5,5)
    ax[0].legend(loc='lower center')
    
    true_phase = nf_field_phase - src_field_phase
    true_phase = (true_phase + 2 * np.pi) % (2 * np.pi)

    #interpolate prj_field phase and src_field phase to the same x values of prj_field
    
    src_field_phase_interp = np.interp(x_prj, x_src, src_field_phase)

    true_phase_ff = prj_field_phase - src_field_phase_interp
    true_phase_ff = (true_phase_ff + 2 * np.pi) % (2 * np.pi)

    #ax[1].plot(x_nf,true_phase, '.b', label="FDTD phase")
    ax[1].plot(xs*1e6,output_field_phase, 'r', label="ASM phase")
    ax[1].plot(xs*1e6,gaussian_phase, '-.g', label="theory phase")
    ax[1].plot(x_prj,true_phase_ff, label="FDTD FF phase")

    ax[1].set_xlim(-5,5)
    ax[1].legend()

    #plt.show()
    plt.savefig("asm_fdtd_comparison.pdf")