import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from simsopt.mhd import Vmec
from simsopt._core.util import Struct
import sys
sys.path.append("../utils")
from constants import ALPHA_PARTICLE_MASS,ALPHA_PARTICLE_CHARGE,FUSION_ALPHA_SPEED_SQUARED 


def vmec_splines(vmec):
    """
    Initialize radial splines for a VMEC equilibrium.
    Args:
        vmec: An instance of :obj:`simsopt.mhd.vmec.Vmec`.
    Returns:
        A structure with the splines as attributes.
    """
    vmec.run()
    results = Struct()
    if vmec.wout.lasym:
        raise ValueError("vmec_splines is not yet set up for non-stellarator-symmetric cases.")

    rmnc = []
    zmns = []
    lmns = []
    d_rmnc_d_s = []
    d_zmns_d_s = []
    d_lmns_d_s = []
    for jmn in range(vmec.wout.mnmax):
        rmnc.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.rmnc[jmn, :]))
        zmns.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.zmns[jmn, :]))
        lmns.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.lmns[jmn, 1:]))
        d_rmnc_d_s.append(rmnc[-1].derivative())
        d_zmns_d_s.append(zmns[-1].derivative())
        d_lmns_d_s.append(lmns[-1].derivative())

    gmnc = []
    bmnc = []
    bsupumnc = []
    bsupvmnc = []
    bsubsmns = []
    bsubumnc = []
    bsubvmnc = []
    d_bmnc_d_s = []
    d_bsupumnc_d_s = []
    d_bsupvmnc_d_s = []
    d_bsubumnc_d_s = [] # new
    d_bsubvmnc_d_s = [] # new
    for jmn in range(vmec.wout.mnmax_nyq):
        gmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.gmnc[jmn, 1:]))
        bmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bmnc[jmn, 1:]))
        bsupumnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupumnc[jmn, 1:]))
        bsupvmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupvmnc[jmn, 1:]))
        # Note that bsubsmns is on the full mesh, unlike the other components:
        bsubsmns.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.bsubsmns[jmn, :]))
        bsubumnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsubumnc[jmn, 1:]))
        bsubvmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsubvmnc[jmn, 1:]))
        d_bmnc_d_s.append(bmnc[-1].derivative())
        d_bsupumnc_d_s.append(bsupumnc[-1].derivative())
        d_bsupvmnc_d_s.append(bsupvmnc[-1].derivative())
        d_bsubumnc_d_s.append(bsubumnc[-1].derivative()) # new
        d_bsubvmnc_d_s.append(bsubvmnc[-1].derivative()) # new

    # Handle 1d profiles:
    results.pressure = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.pres[1:])
    results.d_pressure_d_s = results.pressure.derivative()
    results.iota = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:])
    results.d_iota_d_s = results.iota.derivative()

    # Save other useful quantities:
    results.phiedge = vmec.wout.phi[-1]
    variables = ['Aminor_p', 'mnmax', 'xm', 'xn', 'mnmax_nyq', 'xm_nyq', 'xn_nyq', 'nfp']
    for v in variables:
        results.__setattr__(v, eval('vmec.wout.' + v))

    variables = ['rmnc', 'zmns', 'lmns', 'd_rmnc_d_s', 'd_zmns_d_s', 'd_lmns_d_s',
                 'gmnc', 'bmnc', 'd_bmnc_d_s', 'bsupumnc', 'bsupvmnc', 'd_bsupumnc_d_s', 'd_bsupvmnc_d_s',
                 'd_bsubumnc_d_s', 'd_bsubvmnc_d_s', # new
                 'bsubsmns', 'bsubumnc', 'bsubvmnc']
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


def vmec_compute_geometry(vs, s, theta, phi, phi_center=0):
    r"""
    Compute many geometric quantities of interest from a vmec configuration.
    Some of the quantities computed by this function refer to
    ``alpha``, a field line label coordinate defined by
    .. math::
        \alpha = \theta_{pest} - \iota (\phi - \phi_{center}).
    Here, :math:`\phi_{center}` is a constant, usually 0, which can be
    set to a nonzero value if desired so the magnetic shear
    contribution to :math:`\nabla\alpha` vanishes at a toroidal angle
    different than 0.  Also, wherever the term ``psi`` appears in
    variable names in this function and the returned arrays, it means
    :math:`\psi =` the toroidal flux divided by :math:`2\pi`, so
    .. math::
        \vec{B} = \nabla\psi\times\nabla\theta_{pest} + \iota\nabla\phi\times\nabla\psi = \nabla\psi\times\nabla\alpha.
    Most of the arrays that are returned by this function have shape
    ``(ns, ntheta, nphi)``, where ``ns`` is the number of flux
    surfaces, ``ntheta`` is the number of grid points in VMEC's
    poloidal angle, and ``nphi`` is the number of grid points in the
    standard toroidal angle. For the arguments ``theta`` and ``phi``,
    you can either provide 1D arrays, in which case a tensor product
    grid is used, or you can provide 3D arrays of shape ``(ns, ntheta,
    nphi)``. In this latter case, the grids are not necessarily
    tensor-product grids.  Note that all angles in this function have
    period :math:`2\pi`, not period 1.
    The output arrays are returned as attributes of the
    returned object. Many intermediate quantities are included, such
    as the Cartesian components of the covariant and contravariant
    basis vectors. Some of the most useful of these output arrays are (all with SI units):
    - ``phi``: The standard toroidal angle :math:`\phi`.
    - ``theta_vmec``: VMEC's poloidal angle :math:`\theta_{vmec}`.
    - ``theta_pest``: The straight-field-line angle :math:`\theta_{pest}` associated with :math:`\phi`.
    - ``modB``: The magnetic field magnitude :math:`|B|`.
    - ``B_sup_theta_vmec``: :math:`\vec{B}\cdot\nabla\theta_{vmec}`.
    - ``B_sup_phi``: :math:`\vec{B}\cdot\nabla\phi`.
    - ``B_cross_grad_B_dot_grad_alpha``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\alpha`.
    - ``B_cross_grad_B_dot_grad_psi``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\psi`.
    - ``B_cross_kappa_dot_grad_alpha``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\alpha`,
      where :math:`\vec{\kappa}=\vec{b}\cdot\nabla\vec{b}` is the curvature and :math:`\vec{b}=|B|^{-1}\vec{B}`.
    - ``B_cross_kappa_dot_grad_psi``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\psi`.
    - ``grad_alpha_dot_grad_alpha``: :math:`|\nabla\alpha|^2 = \nabla\alpha\cdot\nabla\alpha`.
    - ``grad_alpha_dot_grad_psi``: :math:`\nabla\alpha\cdot\nabla\psi`.
    - ``grad_psi_dot_grad_psi``: :math:`|\nabla\psi|^2 = \nabla\psi\cdot\nabla\psi`.
    - ``iota``: The rotational transform :math:`\iota`. This array has shape ``(ns,)``.
    - ``shat``: The magnetic shear :math:`\hat s= (x/q) (d q / d x)` where 
      :math:`x = \mathrm{Aminor_p} \, \sqrt{s}` and :math:`q=1/\iota`. This array has shape ``(ns,)``.
    The following normalized versions of these quantities used in the
    gyrokinetic codes ``stella``, ``gs2``, and ``GX`` are also
    returned: ``bmag``, ``gbdrift``, ``gbdrift0``, ``cvdrift``,
    ``cvdrift0``, ``gds2``, ``gds21``, and ``gds22``, along with
    ``L_reference`` and ``B_reference``.  Instead of ``gradpar``, two
    variants are returned, ``gradpar_theta_pest`` and ``gradpar_phi``,
    corresponding to choosing either :math:`\theta_{pest}` or
    :math:`\phi` as the parallel coordinate.
    The value(s) of ``s`` provided as input need not coincide with the
    full grid or half grid in VMEC, as spline interpolation will be
    used radially.
    The implementation in this routine is similar to the one in the
    gyrokinetic code ``stella``.
    Example usage::
        import numpy as np
        from simsopt.mhd import Vmec, vmec_compute_geometry
        v = Vmec("wout_li383_1.4m.nc")
        s = 1
        theta = np.linspace(0, 2 * np.pi, 50)
        phi = np.linspace(0, 2 * np.pi / 3, 60)
        data = vmec_compute_geometry(v, s, theta, phi)
        print(data.grad_s_dot_grad_s)
    Args:
        vs: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`
          or the structure returned by :func:`vmec_splines`.
        s: Values of normalized toroidal flux on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        theta: Values of vmec's poloidal angle. You can provide a float, a 1d array of size
          ``(ntheta,)``, or a 3d array of size ``(ns, ntheta, nphi)``.
        phi: Values of the standard toroidal angle. You can provide a float, a 1d array of size
          ``(nphi,)`` or a 3d array of size ``(ns, ntheta, nphi)``.
        phi_center: :math:`\phi_{center}`, an optional shift to the toroidal angle
          in the definition of :math:`\alpha`.
    """
    # If given a Vmec object, convert it to vmec_splines:
    if isinstance(vs, Vmec):
        vs = vmec_splines(vs)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Handle theta
    try:
        ntheta = len(theta)
    except:
        theta = [theta]
    theta_vmec = np.array(theta)
    if theta_vmec.ndim == 1:
        ntheta = len(theta_vmec)
    elif theta_vmec.ndim == 3:
        ntheta = theta_vmec.shape[1]
    else:
        raise ValueError("theta argument must be a float, 1d array, or 3d array.")

    # Handle phi
    try:
        nphi = len(phi)
    except:
        phi = [phi]
    phi = np.array(phi)
    if phi.ndim == 1:
        nphi = len(phi)
    elif phi.ndim == 3:
        nphi = phi.shape[2]
    else:
        raise ValueError("phi argument must be a float, 1d array, or 3d array.")

    # If theta and phi are not already 3D, make them 3D:
    if theta_vmec.ndim == 1:
        theta_vmec = np.kron(np.ones((ns, 1, nphi)), theta_vmec.reshape(1, ntheta, 1))
    if phi.ndim == 1:
        phi = np.kron(np.ones((ns, ntheta, 1)), phi.reshape(1, 1, nphi))

    # Shorthand:
    mnmax = vs.mnmax
    xm = vs.xm
    xn = vs.xn
    mnmax_nyq = vs.mnmax_nyq
    xm_nyq = vs.xm_nyq
    xn_nyq = vs.xn_nyq

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    # shat = (r/q)(dq/dr) where r = a sqrt(s)
    #      = - (r/iota) (d iota / d r) = -2 (s/iota) (d iota / d s)
    shat = (-2 * s / iota) * d_iota_d_s

    rmnc = np.zeros((ns, mnmax))
    zmns = np.zeros((ns, mnmax))
    lmns = np.zeros((ns, mnmax))
    d_rmnc_d_s = np.zeros((ns, mnmax))
    d_zmns_d_s = np.zeros((ns, mnmax))
    d_lmns_d_s = np.zeros((ns, mnmax))
    for jmn in range(mnmax):
        rmnc[:, jmn] = vs.rmnc[jmn](s)
        zmns[:, jmn] = vs.zmns[jmn](s)
        lmns[:, jmn] = vs.lmns[jmn](s)
        d_rmnc_d_s[:, jmn] = vs.d_rmnc_d_s[jmn](s)
        d_zmns_d_s[:, jmn] = vs.d_zmns_d_s[jmn](s)
        d_lmns_d_s[:, jmn] = vs.d_lmns_d_s[jmn](s)

    gmnc = np.zeros((ns, mnmax_nyq))
    bmnc = np.zeros((ns, mnmax_nyq))
    d_bmnc_d_s = np.zeros((ns, mnmax_nyq))
    bsupumnc = np.zeros((ns, mnmax_nyq))
    bsupvmnc = np.zeros((ns, mnmax_nyq))
    bsubsmns = np.zeros((ns, mnmax_nyq))
    bsubumnc = np.zeros((ns, mnmax_nyq))
    bsubvmnc = np.zeros((ns, mnmax_nyq))
    d_bsubumnc_d_s = np.zeros((ns, mnmax_nyq)) # new
    d_bsubvmnc_d_s = np.zeros((ns, mnmax_nyq)) # new
    for jmn in range(mnmax_nyq):
        gmnc[:, jmn] = vs.gmnc[jmn](s)
        bmnc[:, jmn] = vs.bmnc[jmn](s)
        d_bmnc_d_s[:, jmn] = vs.d_bmnc_d_s[jmn](s)
        bsupumnc[:, jmn] = vs.bsupumnc[jmn](s)
        bsupvmnc[:, jmn] = vs.bsupvmnc[jmn](s)
        bsubsmns[:, jmn] = vs.bsubsmns[jmn](s)
        bsubumnc[:, jmn] = vs.bsubumnc[jmn](s)
        bsubvmnc[:, jmn] = vs.bsubvmnc[jmn](s)
        d_bsubumnc_d_s[:, jmn] = vs.d_bsubumnc_d_s[jmn](s)  # new
        d_bsubvmnc_d_s[:, jmn] = vs.d_bsubvmnc_d_s[jmn](s)  # new

    # Now that we know theta_vmec, compute all the geometric quantities
    angle = xm[:, None, None, None] * theta_vmec[None, :, :, :] - xn[:, None, None, None] * phi[None, :, :, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm[:, None, None, None] * cosangle
    ncosangle = xn[:, None, None, None] * cosangle
    msinangle = xm[:, None, None, None] * sinangle
    nsinangle = xn[:, None, None, None] * sinangle
    # Order of indices in cosangle and sinangle: mn, s, theta, phi
    # Order of indices in rmnc, bmnc, etc: s, mn
    R = np.einsum('ij,jikl->ikl', rmnc, cosangle)
    d_R_d_s = np.einsum('ij,jikl->ikl', d_rmnc_d_s, cosangle)
    d_R_d_theta_vmec = -np.einsum('ij,jikl->ikl', rmnc, msinangle)
    d_R_d_phi = np.einsum('ij,jikl->ikl', rmnc, nsinangle)

    Z = np.einsum('ij,jikl->ikl', zmns, sinangle)
    d_Z_d_s = np.einsum('ij,jikl->ikl', d_zmns_d_s, sinangle)
    d_Z_d_theta_vmec = np.einsum('ij,jikl->ikl', zmns, mcosangle)
    d_Z_d_phi = -np.einsum('ij,jikl->ikl', zmns, ncosangle)

    lambd = np.einsum('ij,jikl->ikl', lmns, sinangle)
    d_lambda_d_s = np.einsum('ij,jikl->ikl', d_lmns_d_s, sinangle)
    d_lambda_d_theta_vmec = np.einsum('ij,jikl->ikl', lmns, mcosangle)
    d_lambda_d_phi = -np.einsum('ij,jikl->ikl', lmns, ncosangle)
    theta_pest = theta_vmec + lambd

    # Now handle the Nyquist quantities:
    angle = xm_nyq[:, None, None, None] * theta_vmec[None, :, :, :] - xn_nyq[:, None, None, None] * phi[None, :, :, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm_nyq[:, None, None, None] * cosangle
    ncosangle = xn_nyq[:, None, None, None] * cosangle
    msinangle = xm_nyq[:, None, None, None] * sinangle
    nsinangle = xn_nyq[:, None, None, None] * sinangle

    sqrt_g_vmec = np.einsum('ij,jikl->ikl', gmnc, cosangle)
    modB = np.einsum('ij,jikl->ikl', bmnc, cosangle)
    d_B_d_s = np.einsum('ij,jikl->ikl', d_bmnc_d_s, cosangle)
    d_B_d_theta_vmec = -np.einsum('ij,jikl->ikl', bmnc, msinangle)
    d_B_d_phi = np.einsum('ij,jikl->ikl', bmnc, nsinangle)

    B_sup_theta_vmec = np.einsum('ij,jikl->ikl', bsupumnc, cosangle)
    B_sup_phi = np.einsum('ij,jikl->ikl', bsupvmnc, cosangle)
    B_sub_s = np.einsum('ij,jikl->ikl', bsubsmns, sinangle)
    B_sub_theta_vmec = np.einsum('ij,jikl->ikl', bsubumnc, cosangle)
    B_sub_phi = np.einsum('ij,jikl->ikl', bsubvmnc, cosangle)
    B_sup_theta_pest = iota[:, None, None] * B_sup_phi

    # covariant derivs; new block
    d_B_sub_s_d_theta_vmec = np.einsum('ij,jikl->ikl', bsubsmns, mcosangle)
    d_B_sub_s_d_phi = -np.einsum('ij,jikl->ikl', bsubsmns, ncosangle)
    d_B_sub_theta_vmec_d_s = np.einsum('ij,jikl->ikl', d_bsubumnc_d_s, cosangle)
    d_B_sub_theta_vmec_d_phi = np.einsum('ij,jikl->ikl', bsubumnc, nsinangle)
    d_B_sub_phi_d_s = np.einsum('ij,jikl->ikl', d_bsubvmnc_d_s, cosangle)
    d_B_sub_phi_d_theta_vmec = -np.einsum('ij,jikl->ikl', bsubvmnc,msinangle)

    sqrt_g_vmec_alt = R * (d_Z_d_s * d_R_d_theta_vmec - d_R_d_s * d_Z_d_theta_vmec)

    # Note the minus sign. psi in the straight-field-line relation seems to have opposite sign to vmec's phi array.
    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)

    # *********************************************************************
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # *********************************************************************
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    X = R * cosphi
    d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
    d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
    d_X_d_s = d_R_d_s * cosphi
    Y = R * sinphi
    d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi
    d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
    d_Y_d_s = d_R_d_s * sinphi

    # Now use the dual relations to get the Cartesian components of grad s, grad theta_vmec, and grad phi:
    grad_s_X = (d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi) / sqrt_g_vmec
    grad_s_Y = (d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi) / sqrt_g_vmec
    grad_s_Z = (d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi) / sqrt_g_vmec

    grad_theta_vmec_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
    grad_theta_vmec_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
    grad_theta_vmec_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

    grad_phi_X = (d_Y_d_s * d_Z_d_theta_vmec - d_Z_d_s * d_Y_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Y = (d_Z_d_s * d_X_d_theta_vmec - d_X_d_s * d_Z_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Z = (d_X_d_s * d_Y_d_theta_vmec - d_Y_d_s * d_X_d_theta_vmec) / sqrt_g_vmec
    # End of dual relations.

    # *********************************************************************
    # Compute the Cartesian components of other quantities we need:
    # *********************************************************************

    grad_psi_X = grad_s_X * edge_toroidal_flux_over_2pi
    grad_psi_Y = grad_s_Y * edge_toroidal_flux_over_2pi
    grad_psi_Z = grad_s_Z * edge_toroidal_flux_over_2pi

    # Form grad alpha = grad (theta_vmec + lambda - iota * phi)
    grad_alpha_X = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_X
    grad_alpha_Y = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_Y
    grad_alpha_Z = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_Z

    grad_alpha_X += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_X + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_X
    grad_alpha_Y += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Y + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_Y
    grad_alpha_Z += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Z + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_Z

    grad_B_X = d_B_d_s * grad_s_X + d_B_d_theta_vmec * grad_theta_vmec_X + d_B_d_phi * grad_phi_X
    grad_B_Y = d_B_d_s * grad_s_Y + d_B_d_theta_vmec * grad_theta_vmec_Y + d_B_d_phi * grad_phi_Y
    grad_B_Z = d_B_d_s * grad_s_Z + d_B_d_theta_vmec * grad_theta_vmec_Z + d_B_d_phi * grad_phi_Z

    B_X = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_X_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_X_d_theta_vmec) / sqrt_g_vmec
    B_Y = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_Y_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_Y_d_theta_vmec) / sqrt_g_vmec
    B_Z = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_Z_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_Z_d_theta_vmec) / sqrt_g_vmec

    # *********************************************************************
    # For gbdrift, we need \vect{B} cross grad |B| dot grad alpha.
    # For cvdrift, we also need \vect{B} cross grad s dot grad alpha.
    # Let us compute both of these quantities 2 ways, and make sure the two
    # approaches give the same answer (within some tolerance).
    # *********************************************************************

    B_cross_grad_s_dot_grad_alpha = (B_sub_phi * (1 + d_lambda_d_theta_vmec) \
                                     - B_sub_theta_vmec * (d_lambda_d_phi - iota[:, None, None])) / sqrt_g_vmec

    B_cross_grad_s_dot_grad_alpha_alternate = 0 \
        + B_X * grad_s_Y * grad_alpha_Z \
        + B_Y * grad_s_Z * grad_alpha_X \
        + B_Z * grad_s_X * grad_alpha_Y \
        - B_Z * grad_s_Y * grad_alpha_X \
        - B_X * grad_s_Z * grad_alpha_Y \
        - B_Y * grad_s_X * grad_alpha_Z

    B_cross_grad_B_dot_grad_alpha = 0 \
        + (B_sub_s * d_B_d_theta_vmec * (d_lambda_d_phi - iota[:, None, None]) \
           + B_sub_theta_vmec * d_B_d_phi * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) \
           + B_sub_phi * d_B_d_s * (1 + d_lambda_d_theta_vmec) \
           - B_sub_phi * d_B_d_theta_vmec * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) \
           - B_sub_theta_vmec * d_B_d_s * (d_lambda_d_phi - iota[:, None, None]) \
           - B_sub_s * d_B_d_phi * (1 + d_lambda_d_theta_vmec)) / sqrt_g_vmec

    B_cross_grad_B_dot_grad_alpha_alternate = 0 \
        + B_X * grad_B_Y * grad_alpha_Z \
        + B_Y * grad_B_Z * grad_alpha_X \
        + B_Z * grad_B_X * grad_alpha_Y \
        - B_Z * grad_B_Y * grad_alpha_X \
        - B_X * grad_B_Z * grad_alpha_Y \
        - B_Y * grad_B_X * grad_alpha_Z

    grad_alpha_dot_grad_alpha = grad_alpha_X * grad_alpha_X + grad_alpha_Y * grad_alpha_Y + grad_alpha_Z * grad_alpha_Z

    grad_alpha_dot_grad_psi = grad_alpha_X * grad_psi_X + grad_alpha_Y * grad_psi_Y + grad_alpha_Z * grad_psi_Z

    grad_psi_dot_grad_psi = grad_psi_X * grad_psi_X + grad_psi_Y * grad_psi_Y + grad_psi_Z * grad_psi_Z

    grad_s_dot_grad_s = grad_s_X * grad_s_X + grad_s_Y * grad_s_Y + grad_s_Z * grad_s_Z

    B_cross_grad_B_dot_grad_psi = (B_sub_theta_vmec * d_B_d_phi - B_sub_phi * d_B_d_theta_vmec) / sqrt_g_vmec * edge_toroidal_flux_over_2pi

    B_cross_kappa_dot_grad_psi = B_cross_grad_B_dot_grad_psi / modB

    mu_0 = 4 * np.pi * (1.0e-7)
    B_cross_kappa_dot_grad_alpha = B_cross_grad_B_dot_grad_alpha / modB + mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi

    # stella / gs2 / gx quantities:

    L_reference = vs.Aminor_p
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    sqrt_s = np.sqrt(s)

    bmag = modB / B_reference

    gradpar_theta_pest = L_reference * B_sup_theta_pest / modB

    gradpar_phi = L_reference * B_sup_phi / modB

    gds2 = grad_alpha_dot_grad_alpha * L_reference * L_reference * s[:, None, None]

    gds21 = grad_alpha_dot_grad_psi * shat[:, None, None] / B_reference

    gds22 = grad_psi_dot_grad_psi * shat[:, None, None] * shat[:, None, None] / (L_reference * L_reference * B_reference * B_reference * s[:, None, None])

    # temporary fix. Please see issue #238 and the discussion therein
    gbdrift = -1 * 2 * B_reference * L_reference * L_reference * sqrt_s[:, None, None] * B_cross_grad_B_dot_grad_alpha / (modB * modB * modB) * toroidal_flux_sign

    gbdrift0 = B_cross_grad_B_dot_grad_psi * 2 * shat[:, None, None] / (modB * modB * modB * sqrt_s[:, None, None]) * toroidal_flux_sign

    # temporary fix. Please see issue #238 and the discussion therein
    cvdrift = gbdrift - 2 * B_reference * L_reference * L_reference * sqrt_s[:, None, None] * mu_0 * d_pressure_d_s[:, None, None] * toroidal_flux_sign / (edge_toroidal_flux_over_2pi * modB * modB)

    cvdrift0 = gbdrift0

    # Package results into a structure to return:
    results = Struct()
    variables = ['ns', 'ntheta', 'nphi', 's', 'iota', 'd_iota_d_s', 'd_pressure_d_s', 'shat',
                 'theta_vmec', 'phi', 'theta_pest',
                 'd_lambda_d_s', 'd_lambda_d_theta_vmec', 'd_lambda_d_phi', 'sqrt_g_vmec', 'sqrt_g_vmec_alt',
                 'modB', 'd_B_d_s', 'd_B_d_theta_vmec', 'd_B_d_phi', 'B_sup_theta_vmec', 'B_sup_theta_pest', 'B_sup_phi',
                 'B_sub_s', 'B_sub_theta_vmec', 'B_sub_phi', 'edge_toroidal_flux_over_2pi', 'sinphi', 'cosphi',
                 'd_B_sub_s_d_theta_vmec', # new
                 'd_B_sub_s_d_phi', # new
                 'd_B_sub_theta_vmec_d_s', # new
                 'd_B_sub_theta_vmec_d_phi', # new
                 'd_B_sub_phi_d_s', # new
                 'd_B_sub_phi_d_theta_vmec', # new
                 'R', 'd_R_d_s', 'd_R_d_theta_vmec', 'd_R_d_phi', 'X', 'Y', 'Z', 'd_Z_d_s', 'd_Z_d_theta_vmec', 'd_Z_d_phi',
                 'd_X_d_theta_vmec', 'd_X_d_phi', 'd_X_d_s', 'd_Y_d_theta_vmec', 'd_Y_d_phi', 'd_Y_d_s',
                 'grad_s_X', 'grad_s_Y', 'grad_s_Z', 'grad_theta_vmec_X', 'grad_theta_vmec_Y', 'grad_theta_vmec_Z',
                 'grad_phi_X', 'grad_phi_Y', 'grad_phi_Z', 'grad_psi_X', 'grad_psi_Y', 'grad_psi_Z',
                 'grad_alpha_X', 'grad_alpha_Y', 'grad_alpha_Z', 'grad_B_X', 'grad_B_Y', 'grad_B_Z',
                 'B_X', 'B_Y', 'B_Z', "grad_s_dot_grad_s",
                 'B_cross_grad_s_dot_grad_alpha', 'B_cross_grad_s_dot_grad_alpha_alternate',
                 'B_cross_grad_B_dot_grad_alpha', 'B_cross_grad_B_dot_grad_alpha_alternate',
                 'B_cross_grad_B_dot_grad_psi', 'B_cross_kappa_dot_grad_psi', 'B_cross_kappa_dot_grad_alpha',
                 'grad_alpha_dot_grad_alpha', 'grad_alpha_dot_grad_psi', 'grad_psi_dot_grad_psi',
                 'L_reference', 'B_reference', 'toroidal_flux_sign',
                 'bmag', 'gradpar_theta_pest', 'gradpar_phi', 'gds2', 'gds21', 'gds22', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0']
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


def interpolatedVmecField(vmec,s,theta,phi,method='linear'):
    """
    Do 3d interpolation of the fields used for particle tracing in VMEC 
    coords. 
    
    vmec: Vmec object
    s,theta,phi: 1d arrays to build interpolants from.
    method: interpolation method 'linear' or 'cubic'
    return: a struct with interpolants.
    """
    # get vmec data on a 3d mesh
    vmec_data = vmec_compute_geometry(vmec, s, theta, phi)

    # do 3d interpolation
    sqrt_g_vmec = RegularGridInterpolator((s,theta,phi), vmec_data.sqrt_g_vmec,method=method)
    modB = RegularGridInterpolator((s,theta,phi), vmec_data.modB,method=method)
    d_B_d_s = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_d_s,method=method)
    d_B_d_theta_vmec = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_d_theta_vmec,method=method)
    d_B_d_phi = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_d_phi,method=method)
    B_sup_theta_vmec = RegularGridInterpolator((s,theta,phi), vmec_data.B_sup_theta_vmec,method=method)
    B_sup_phi = RegularGridInterpolator((s,theta,phi), vmec_data.B_sup_phi,method=method)
    B_sub_s = RegularGridInterpolator((s,theta,phi), vmec_data.B_sub_s,method=method)
    B_sub_theta_vmec = RegularGridInterpolator((s,theta,phi), vmec_data.B_sub_theta_vmec,method=method)
    B_sub_phi = RegularGridInterpolator((s,theta,phi), vmec_data.B_sub_phi,method=method)
    d_B_sub_s_d_theta_vmec = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_sub_s_d_theta_vmec,method=method)
    d_B_sub_s_d_phi = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_sub_s_d_phi,method=method)
    d_B_sub_theta_vmec_d_s = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_sub_theta_vmec_d_s,method=method)
    d_B_sub_theta_vmec_d_phi = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_sub_theta_vmec_d_phi,method=method)
    d_B_sub_phi_d_s = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_sub_phi_d_s,method=method)
    d_B_sub_phi_d_theta_vmec = RegularGridInterpolator((s,theta,phi), vmec_data.d_B_sub_phi_d_theta_vmec,method=method)

    results = Struct()
    variables = [
        'sqrt_g_vmec','modB',
        'd_B_d_s', 'd_B_d_theta_vmec', 'd_B_d_phi', 
        'B_sup_theta_vmec','B_sup_phi',
        'B_sub_s', 'B_sup_theta_vmec',  'B_sup_phi',
        'B_sub_theta_vmec', 'B_sub_phi', 
        'd_B_sub_s_d_theta_vmec', 'd_B_sub_s_d_phi', 
        'd_B_sub_theta_vmec_d_s', 'd_B_sub_theta_vmec_d_phi',
        'd_B_sub_phi_d_s', 'd_B_sub_phi_d_theta_vmec'
        ]
    for v in variables:
        results.__setattr__(v, eval(v))
    return results


class GuidingCenterVmec:
    """
    A guiding center object for tracing a single particle.
    """
   
    def __init__(self,interp,y0):
        self.interp = interp
        self.mass =ALPHA_PARTICLE_MASS
        self.charge=ALPHA_PARTICLE_CHARGE
        # compute mu = vperp^2/2B
        stz   = y0[:3]
        v_par = y0[3]
        modB = interp.modB(stz).flatten()
        vperp_squared = FUSION_ALPHA_SPEED_SQUARED - v_par**2
        self.mu = vperp_squared/2/modB

    # TODO: write a GuidingCenter Vacumm RHS

    def GuidingCenterVmecRHS(self,ys):
        #TODO: use field period symmetry and stellarator symmetry
        # to map particles back to interpolation range.
        """
        Guiding center right hand side for vmec tracing.
        ys: (4,) array [s,t,z,vpar]
        """
        # struct with interpolators
        interp = self.interp

        # unpack point
        stp = ys[:3]
        vpar = ys[-1]

        # compute the values for the rhs
        mu = self.mu
        sqrt_g_vmec = interp.sqrt_g_vmec(stp).item()
        modB = interp.modB(stp).item()
        d_B_d_s = interp.d_B_d_s(stp).item()
        d_B_d_theta_vmec = interp.d_B_d_theta_vmec(stp).item()
        d_B_d_phi = interp.d_B_d_phi(stp).item()
        B_sup_theta_vmec = interp.B_sup_theta_vmec(stp).item()
        B_sup_phi = interp.B_sup_phi(stp).item()
        B_sub_s = interp.B_sub_s(stp).item()
        B_sub_theta_vmec = interp.B_sub_theta_vmec(stp).item()
        B_sub_phi = interp.B_sub_phi(stp).item()
        d_B_sub_s_d_theta_vmec = interp.d_B_sub_s_d_theta_vmec(stp).item()
        d_B_sub_s_d_phi = interp.d_B_sub_s_d_phi(stp).item()
        d_B_sub_theta_vmec_d_s = interp.d_B_sub_theta_vmec_d_s(stp).item()
        d_B_sub_theta_vmec_d_phi = interp.d_B_sub_theta_vmec_d_phi(stp).item()
        d_B_sub_phi_d_s = interp.d_B_sub_phi_d_s(stp).item()
        d_B_sub_phi_d_theta_vmec = interp.d_B_sub_phi_d_theta_vmec(stp).item()
        Omega = self.charge*modB/self.mass

        # vpar**2
        vp_sq = vpar**2

        # ds/dt
        coeff = (1.0/modB)*(1.0/Omega/sqrt_g_vmec)
        a1 = -mu*B_sub_phi*d_B_d_theta_vmec
        a2 = mu*B_sub_theta_vmec*d_B_d_phi
        a3 = vp_sq*d_B_sub_phi_d_theta_vmec
        a4 = -vp_sq*B_sub_phi*d_B_d_theta_vmec/modB
        a5 = -vp_sq*d_B_sub_theta_vmec_d_phi
        a6 = vp_sq*B_sub_theta_vmec*d_B_d_phi/modB
        d_s_d_t = coeff*(a1+a2+a3+a4+a5+a6)

        # dtheta/dt
        a1 = vpar*B_sup_theta_vmec/modB
        a2 = mu*B_sub_phi*d_B_d_s
        a3 = -mu*B_sub_s*d_B_d_phi
        a4 = -vp_sq*d_B_sub_phi_d_s
        a5 = (vp_sq/modB)*B_sub_phi*d_B_d_s
        a6 = vp_sq*d_B_sub_s_d_phi
        a7 = -(vp_sq/modB)*B_sub_s*d_B_d_phi
        d_theta_vmec_d_t = a1 + coeff*(a2+a3+a4+a5+a6+a7)
 
        # dphi/dt
        a1 = vpar*B_sup_phi/modB
        a2 = -mu*B_sub_theta_vmec*d_B_d_s
        a3 = mu*B_sub_s*d_B_d_theta_vmec
        a4 = vp_sq*d_B_sub_theta_vmec_d_s
        a5 = -(vp_sq/modB)*B_sub_theta_vmec*d_B_d_s
        a6 = -vp_sq*d_B_sub_s_d_theta_vmec
        a7 = (vp_sq/modB)*B_sub_s*d_B_d_theta_vmec
        d_phi_d_t = a1 + coeff*(a2+a3+a4+a5+a6+a7)

        # dvpar/dt
        a1 = -(mu/modB)*(B_sup_theta_vmec*d_B_d_theta_vmec + B_sup_phi*d_B_d_phi)
        coeff = -mu*vpar/modB/Omega/sqrt_g_vmec
        a2 = d_B_d_s*d_B_sub_phi_d_theta_vmec
        a3 = -d_B_d_s*d_B_sub_theta_vmec_d_phi
        a4 = -d_B_d_theta_vmec*d_B_sub_phi_d_s
        a5 = d_B_d_theta_vmec*d_B_sub_s_d_phi
        a6 = d_B_d_phi*d_B_sub_theta_vmec_d_s
        a7 = -d_B_d_phi*d_B_sub_s_d_theta_vmec
        d_vpar_d_t = a1 + coeff*(a2+a3+a4+a5+a6+a7)

        return np.array([d_s_d_t,d_theta_vmec_d_t,d_phi_d_t,d_vpar_d_t])
        

def trace_particles_vmec(interp,
          stp_inits, 
          vpar_inits, 
          tmax=1e-4, 
          atol=1e-6,
          min_step=0.0,
          comm=None,
          stopping_criteria=None):
    # TODO: set up the mpi comms

    n_particles = len(stp_inits)

    # storage
    res_traj = []
    res_events = []
    for ii,stp in enumerate(stp_inits):
      # get the initial point
      vp = vpar_inits[ii]
      y0 = np.append(stp,vp)
      # define the guiding center eqns
      gc_rhs = GuidingCenterVmec(interp,y0).GuidingCenterVmecRHS
      # solve the ode
      res = solve_ivp(gc_rhs,(0,tmax),y0,events=stopping_criteria,atol=atol,min_step=min_step)
      # now unpack the results
      res_traj.append(np.vstack((res.t,res.y)).T) # trajectories [[t,s,theta,phi],...]
      res_events.append([res.t_events,res.y_events]) 

    return res_traj,res_events



# TODO: generalize this stopping criteria
def MaxStoppingCriteria(t,y):
    """
    Finds a zero of g(s) = s - 1.
    """
    return y[0] - 1.0
MaxStoppingCriteria.direction = 1.0
MaxStoppingCriteria.terminal = True

# TODO: generalize this stopping criteria
def MinStoppingCriteria(t,y):
    """
    Finds a zero of g(s) = s - 1.
    """
    return y[0] - 0.01
MinStoppingCriteria.direction = -1.0
MinStoppingCriteria.terminal = True


if __name__ == "__main__":
  from simsopt.util.mpi import MpiPartition
  from constants import V_MAX
  vmec_input = '../vmec_input_files/input.nfp2_QA_cold_high_res'
  mpi = MpiPartition()
  vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
  surf = vmec.boundary

  # TODO: need to shut up the elements in vmec_compute_geometry that
  # complain when we evaluate at s=0
  # TODO: exploit stellsym and field period symmetry in interpolants

  # build 3d interpolants
  s = np.linspace(0,1, 16)
  theta = np.linspace(0, 2*np.pi, 32)
  phi = np.linspace(0,2*np.pi/surf.nfp, 32)
  vmec_data = vmec_compute_geometry(vmec, s, theta, phi)
  interp = interpolatedVmecField(vmec,s,theta,phi,'cubic')

  y0 = np.array([0.5,np.pi/2,np.pi/2,V_MAX/2])
  gc_rhs = GuidingCenterVmec(interp,y0).GuidingCenterVmecRHS
  print(gc_rhs(y0))

  quit()

  # use fixed particle locations
  ns = ntheta=nphi=3
  nvpar=3
  surfaces = np.linspace(0,1, ns)
  thetas = np.linspace(0, 2*np.pi, ntheta)
  phis = np.linspace(0,2*np.pi/surf.nfp, nphi)
  vpars = np.linspace(-V_MAX,V_MAX,nvpar)
  # build a mesh
  [surfaces,thetas,phis,vpars] = np.meshgrid(surfaces,thetas,phis,vpars)
  stp_inits = np.zeros((ns*ntheta*nphi*nvpar, 3))
  stp_inits[:, 0] = surfaces.flatten()
  stp_inits[:, 1] = thetas.flatten()
  stp_inits[:, 2] = phis.flatten()
  vpar_inits = vpars.flatten()
  tmax = 1e-4
  # stopping criteria
  stopping_criteria = [MinStoppingCriteria,MinStoppingCriteria]
  # trace
  res_traj, res_events = trace_particles_vmec(interp,
          stp_inits, 
          vpar_inits, 
          tmax=tmax, 
          atol=1e-6,
          min_step=0.0,
          comm=None,
          stopping_criteria=stopping_criteria)

  print(res_events)
  print(res_events.shape)
