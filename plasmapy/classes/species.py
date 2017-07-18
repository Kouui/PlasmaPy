"""
Class representing a group of particles"""
# coding=utf-8
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from astropy.constants import c
from astropy import units as u


class Species:
    """
    Object representing a species of particles: ions, electrons, or simply
    a group of particles with a particular initial velocity distribution.

    Parameters
    ----------
    q : float
        particle charge
    m : float
        particle mass
    scaling : float
        number of particles represented by each macroparticle
    pusher : function
        particle push algorithm
    name : str
        name of group
    """

    def __init__(self, plasma, q = 1 * u.C, m = 1 * u.kg, n = 1, scaling=1, dt=np.inf, nt=np.inf,
                 name="particles"):
        self.q = q
        assert self.q.si.unit == u.C
        self.m = m
        assert self.m.si.unit == u.kg
        self.N = int(n)
        self.scaling = scaling
        self.eff_q = q * scaling
        assert self.eff_q.si.unit == u.C
        self.eff_m = m * scaling
        assert self.eff_m.si.unit == u.kg

        self.plasma = plasma
        if np.isinf(dt) and np.isinf(nt):
            raise ValueError("Both dt and nt are infinite.")

        self.dt = dt
        assert self.dt.si.unit == u.s
        self.NT = nt
        self.t = np.arange(nt) * dt

        self.x = np.zeros((n, 3), dtype=float) * u.m
        self.v = np.zeros((n, 3), dtype=float) * (u.m / u.s)
        self.name = name

        self.position_history = np.zeros((self.NT, *self.x.shape),
                                         dtype=float) * u.m
        self.velocity_history = np.zeros((self.NT, *self.v.shape),
                                         dtype=float) * (u.m / u.s)
        self.B_interpolator = RegularGridInterpolator(
            (self.plasma.x.si.value,
             self.plasma.y.si.value,
             self.plasma.z.si.value),
            self.plasma.magnetic_field.T.si.value,
            method="linear",
            bounds_error=True)

    def interpolate_fields(self):
        interpolated_b = self.B_interpolator(self.x.si.value) * u.T
        # TODO: remove this placeholder once we figure out what to do with
        interpolated_e = np.zeros(interpolated_b.shape) * u.V / u.m
        # local electric fields
        return interpolated_b, interpolated_e

    def boris_push(self, init=False):
        dt = -self.dt / 2 if init else self.dt
        b, e = self.interpolate_fields()

        vminus = self.v + self.eff_q * e / self.eff_m * dt * 0.5

        # rotate to add magnetic field
        t = -b * self.eff_q / self.eff_m * dt * 0.5
        s = 2 * t / (1 + t * t)

        vprime = vminus + np.cross(vminus.si.value, t) * u.m / u.s
        vplus = vminus + np.cross(vprime.si.value, s) * u.m / u.s
        v_new = vplus + self.eff_q * e / self.eff_m * dt * 0.5

        self.v = v_new
        if not init:
            self.x += self.v * dt

    def run(self):
        self.boris_push(init=True)
        for i in range(self.NT):
            self.boris_push()
            self.position_history[i] = self.x
            self.velocity_history[i] = self.v

    def __repr__(self, *args, **kwargs):
        return f"Species(q={self.q:.4f},m={self.m:.4f},N={self.N}," \
               f"name=\"{self.name}\",NT={self.NT})"

    def __str__(self):
        return f"{self.N} {self.scaling:.2e}-{self.name} with " \
               f"q = {self.q:.2e}, m = {self.m:.2e}, " \
               f"{self.saved_iterations} saved history " \
               f"steps over {self.NT} iterations"
