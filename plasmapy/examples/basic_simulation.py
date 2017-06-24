import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from plasmapy import Plasma
import os


def gaussian(x, mean=0.0, std=1.0, amp=1.0):
    """Simple function to return a Gaussian distribution"""
    if isinstance(x, list):
        x = np.array(x)
    power = -((x - mean) ** 2.0) / (2.0 * (std ** 2.0))
    f = amp * np.exp(power)
    if amp == 1:
        f = f / max(f)
    return f


def test_mhd_waves():
    """
    """

    # Define simulation grid and coordinates
    print('- Initiating Simulation... ', end='')
    waves = Plasma(domain_x=np.linspace(-0.5, 0.5, 128)*u.m,
                   domain_y=np.linspace(-0.5, 0.5, 128)*u.m,
                   domain_z=np.linspace(0, 1, 1)*u.m)
    grid = waves.domain_shape
    x, y, z = waves.grid
    r = np.sqrt(x**2 + y**2)
    print('Success')

    # Define initial parameter values - only in perturbed component
    print('- Setting initial conditions... ', end='')
    bfield = np.zeros((3, *grid)) * u.T
    density = (gaussian(r.value, std=0.06, amp=0.9) + 0.1) * u.kg / u.m**3
    energy = (gaussian(r.value, std=0.06, amp=0.9) + 0.1) * u.J / u.m**3

    waves.density = density
    waves.energy = energy
    waves.magnetic_field = bfield
    print('Success')

    fig, ax = plt.subplots()
    density_plot = ax.imshow(waves.density.value, cmap='plasma')
    colorbar = fig.colorbar(density_plot)
    title = ax.set_title("")

    savedir = "/home/dominik/PlasmaPy/plasmapy/tests/test_output/"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    iter_step = 30
    iters = 5001
    frames = iters // iter_step
    def animate(i):
        print(f"Drawing frame {i}, iteration {i*iter_step}")
        vmax = waves.density.max().si.value
        vmin = waves.density.min().si.value
        density_plot.set_data(waves.density)
        density_plot.set_clim(vmin, vmax)
        waves.simulate(max_its=iter_step)
        title.set_text(f"Iteration {waves.simulation_physics.current_iteration}/{iters}")
        fig.savefig(f"{savedir}mhd_waves_{i:03d}")
        return [density_plot, title]
    def init():
        density_plot.set_data(waves.density)

        title.set_text(f"{0}, {waves.simulation_physics.current_iteration}")
        fig.savefig(f"{savedir}mhd_waves_{0:03d}")
        return [density_plot, title]

    anim = animation.FuncAnimation(fig, animate, frames, init_func=init, blit=True)
    anim.save(f"{savedir}mhd_waves.mp4", )

if __name__ == '__main__':
    test_mhd_waves()
