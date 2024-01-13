import datetime

import numpy as np

start = datetime.datetime.now()
rng = np.random.default_rng()


def calc_ang_new(rn_ang):
    return np.rad2deg(
        np.arccos(
            (1 / (2 * g))
            * ((1 + g**2) - ((1 - g**2) / (1 + g * (2 * rn_ang - 1))) ** 2)
        )
    )


def calc_transmittence(ang):
    return np.exp(-d_tau / np.abs(np.cos(np.deg2rad(ang))))


def calc_model(cl, ang):
    rn_trans = rng.uniform(0, 1, cl.size)
    rn_abs = rng.uniform(0, 1, cl.size)
    rn_ang = rng.uniform(0, 1, cl.size)
    rn_ang_sign = rng.choice([-1, 1], cl.size)

    trans = calc_transmittence(ang)

    mask_trans = rn_trans < trans
    mask_abs = (rn_abs > alb_scat) & ~mask_trans
    mask_scat = ~mask_trans & ~mask_abs

    ang_scat = ((calc_ang_new(rn_ang) * rn_ang_sign + ang) + 180) % 360 - 180
    ang_new = np.where(mask_scat, ang_scat, ang)

    ang_new = ang_new[~mask_abs]
    cl_new = cl[~mask_abs]

    cl_new += np.where((ang_new >= -90) & (ang_new <= 90), 1, -1)

    rn_refl = rng.uniform(0, 1, cl_new.size)
    mask_refl = (cl_new == surface) & (rn_refl < alb_surface)

    ang_refl = np.where(ang_new > 0, 180 - ang_new, -180 - ang_new)
    ang_new = np.where(mask_refl, ang_refl, ang_new)

    cl_new[mask_refl] -= 1

    refl_toa = np.count_nonzero(cl_new == toa)
    refld_toa.append(refl_toa)
    abs_ground = np.count_nonzero(cl_new == surface)
    absd_surface.append(abs_ground)
    no_scat_strict = np.count_nonzero((cl_new == surface) & (ang_new == ang_sun))
    absd_direct_strict.append(no_scat_strict)
    no_scat_onedeg = np.count_nonzero(
        (cl_new == surface) & ((ang_new >= ang_sun - 1) & (ang_new <= ang_sun + 1))
    )
    abds_direct_onedeg.append(no_scat_onedeg)

    ang_new = ang_new[(cl_new != toa) & (cl_new != surface)]
    cl_new = cl_new[(cl_new != toa) & (cl_new != surface)]

    return cl_new, ang_new


photons = 1_000_000
layers = 20
toa = 0
surface = layers + 1
tau = 10
d_tau = tau / layers
ang_sun = 20
alb_scat = 0.9
g = 0.8
alb_surface = 0.05

refld_toa = []
absd_surface = []
absd_direct_strict = []
abds_direct_onedeg = []

cl = np.ones(photons)
ang = np.full_like(cl, ang_sun)

while cl.size > 0:
    cl, ang = calc_model(cl, ang)

print(f"Scattered at TOA: {sum(refld_toa)}")
print(f"Absorbed in atmosphere: {photons - sum(refld_toa) - sum(absd_surface)}")
print(f"Absorbed at SRF: {sum(absd_surface)}")
print(f"Unscattered photons (strict): {sum(absd_direct_strict)}")
print(f"Unscattered photons (+-1°): {sum(abds_direct_onedeg)}")

end = datetime.datetime.now()
print(f"Run duration: {end - start}")
