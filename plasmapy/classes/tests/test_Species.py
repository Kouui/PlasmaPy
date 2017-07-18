from plasmapy.classes import Plasma, Species
from astropy import units as u
import pytest
import numpy as np
from astropy.modeling import models, fitting



@pytest.fixture()
def uniform_magnetic_field():
    x = np.linspace(-10, 10, 3) * u.m
    test_plasma = Plasma(x, x, x)
    magfieldstr = 1 * u.T
    test_plasma.magnetic_field[2] = magfieldstr
    return test_plasma


def test_particle_uniform_magnetic():
    """
        Tests the particle stepper for a uniform magnetic field motion.
    """
    test_plasma = uniform_magnetic_field()

    q = 1 * u.C
    m = 1 * u.kg
    s = Species(q, m, 1, test_plasma, dt=1e-2 * u.s, nt=int(1e4),
                name="test particle")

    perp_speed = 0.01 * u.m / u.s
    parallel_speed = 1e-5 * u.m / u.s
    s.v[:, 1] = perp_speed

    s.v[:, 2] = parallel_speed
    estimated_gyrofreq = (s.q * test_plasma.magnetic_field_strength.mean()
                          / s.m).to(1 / u.s)
    expected_gyroradius = s.v[0, 1] / estimated_gyrofreq
    estimated_gyroperiod = 2 * np.pi / estimated_gyrofreq

    s.run()

    x = s.position_history[:, 0, 0]
    y = s.position_history[:, 0, 1]
    z = s.position_history[:, 0, 2]

    def plot():
        from astropy.visualization import quantity_support
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        quantity_support()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("")
        ax.plot(x, y, z)
        ax.set_xlabel("$x$ position")
        ax.set_ylabel("$y$ position")
        ax.set_zlabel("$z$ position")
        plt.show()

    p_init = models.Polynomial1D(degree=1)
    fit_p = fitting.LinearLSQFitter()
    p = fit_p(p_init, s.t, z)

    assert np.allclose(z, p(s.t), atol=1e-4 * u.m),\
        "z-velocity doesn't stay constant!"

    estimated_gyroradius = (x.max() - x.min()) / 2
    assert np.isclose(expected_gyroradius, estimated_gyroradius,
                      atol=1e-4 * u.m), "Gyroradii don't match!"

    # plot()
