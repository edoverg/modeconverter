#visualize fields from hdf5 files
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np

input_files = sorted(glob.glob("metapeg_v2-ex-*.h5"))

data = []
times = []

for filename in input_files:
    with h5py.File(filename, "r") as f:
        Ex = f["ex"][:]
        # Extract time from filename (e.g., "metapeg_v2-ex-000050.00.h5" → 50.00)
        time = float(filename.split("-")[-1].split(".h5")[0])
        times.append(time)
        data.append(Ex[:,0,:])
    print(f"Loaded {filename}, time={time}")
print(len(data))
# Create animation
fig, ax = plt.subplots()
x = np.linspace(-4, 4) #assuming x is the last dimension
z = np.linspace(-3, 3) #assuming z is the first dimension
max_clim = np.max([np.abs(data[i]) for i in range(len(data))])
min_clim = np.min([np.abs(data[i]) for i in range(len(data))])
print(max_clim)
def animate(frame):
    ax.clear()

    im = ax.imshow(np.rot90(np.abs(data[frame][:,0,:]),k=3), origin="lower", 
                   extent=[z.min(), z.max(),x.min(), x.max(),],
                   
                   cmap="viridis",
                   vmin=min_clim, vmax=max_clim)
    
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(f"Electric field (Ex) at t={times[frame]:.2f}")
    #plt.colorbar(im, ax=ax, label="Ex (a.u.)")
    return im,

anim = animation.FuncAnimation(fig, animate, frames=len(data), 
                               interval=100, blit=True, repeat=True)
anim.save("results/Ex_animation.gif", writer="ffmpeg", fps=2)
plt.close()
print("Animation saved to results/Ex_animation.gif")

