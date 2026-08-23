import numpy as np
import matplotlib.pyplot as plt

#import the parameter sweep arrays for the central wavelength
diameters = np.load("unit_cell/unit_cell_center_1550_h900nm_res50_wvl_diameters.npy")
print(diameters)
mode_wvl = np.load("unit_cell/unit_cell_center_1550_h900nm_res50_wvl_mode_wvl.npy")
mode_tran_arr = np.load("unit_cell/unit_cell_center_1550_h900nm_res50_wvl_mode_tran.npy")
print(mode_tran_arr)
mode_phase_arr = np.load("unit_cell/unit_cell_center_1550_h900nm_res50_wvl_mode_phase.npy")
print(mode_wvl)
unwrap_mode_phase_arr = np.abs(np.unwrap(mode_phase_arr - mode_phase_arr[0], period=2*np.pi))
#import the reference value array
ref_results = np.loadtxt("unit_cell/results_S21_p700nm_h900nm.txt", skiprows=5,dtype=complex)
R_ref = np.real(ref_results[:,1])
S21_ref = ref_results[:,3]
S21_ref_mag = np.abs(S21_ref)
S21_ref_phase = np.abs(np.unwrap(np.angle(S21_ref) - np.angle(S21_ref[0]), period=2*np.pi))



plt.figure()
plt.subplot(1,2,1)
plt.plot(1e3*diameters/2,mode_tran_arr, '^r-.', label='FDTD')
plt.plot(R_ref,S21_ref_mag, '*k--',label='FEM')
plt.xlabel("Pillar Radius (nm)")
plt.xticks([t for t in np.linspace(1e3*diameters[0]/2, 1e3*diameters[-1]/2, 5)])
plt.gca().set_xticklabels([f"{t:.0f}" for t in np.linspace(1e3*diameters[0]/2, 1e3*diameters[-1]/2, 5)])
plt.ylabel("Transmission")
plt.legend()

plt.subplot(1,2,2)
plt.plot(1e3*diameters/2,unwrap_mode_phase_arr, '^r-.', label='FDTD')
plt.plot(R_ref,S21_ref_phase, '*k--', label='FEM')
plt.xlabel("Pillar Radius (nm)")
plt.xticks([t for t in np.linspace(1e3*diameters[0]/2, 1e3*diameters[-1]/2, 5)])
plt.gca().set_xticklabels([f"{t:.0f}" for t in np.linspace(1e3*diameters[0]/2, 1e3*diameters[-1]/2, 5)])
plt.ylabel("Phase (rad)")
plt.legend()

plt.subplots_adjust(wspace=0.5)
plt.savefig("unit_cell/unit_cell_param_sweep_lineplot.pdf")