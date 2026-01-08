# ============================================================================
# 3D DRONE DYNAMICS SIMULATION
# ============================================================================
# Dette program beregner kræfter, momenter og accelerationer for en drone
# i 3D rummet ved hjælp af Newton-Euler ligninger.
# ============================================================================

import numpy as np
import math as m
from scripts.l_print import lPrint


# ============================================================================
# FUNKTIONER
# ============================================================================

def _output(label, var_name, expr, use_lprint=True):
    """Helper to handle both lPrint and standard print output."""
    if use_lprint:
        lPrint(label, var_name, expr, sp=False)  # Jupyter markdown output
    else:
        print(f'{label}:')
        print(expr)

def derive_rotation_matrix(roll, pitch, yaw, XYZ, useLPrint=True):
    """
    Beregner rotationsmatricen fra lokal til global koordinatsystem.
    
    Rotationsmatricen transformerer vektorer fra dronens lokal koordinatsystem
    til det globale (inertielle) koordinatsystem.
    
    Parameters:
    -----------
    roll : float
        Rotation omkring x-aksen (radianer)
    pitch : float
        Rotation omkring y-aksen (radianer)
    yaw : float
        Rotation omkring z-aksen (radianer)
    XYZ : bool
        True: bruger X-Y-Z konvention (R = Rx @ Ry @ Rz)
        False: bruger Z-Y-X konvention (R = Rz @ Ry @ Rx)
    useLPrint : bool
        If True, use formatted Jupyter output; if False, use plain print
    
    Returns:
    --------
    R : ndarray (3x3)
        Rotationsmatrice
    """
    # Print formulas for rotation matrices
    _output('Rotation omkring x-aksen (roll):', 
           "R_x", 
           "\\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & \\cos(\\phi) & -\\sin(\\phi) \\\\ 0 & \\sin(\\phi) & \\cos(\\phi) \\end{bmatrix}",
           useLPrint)
    
    _output('Rotation omkring y-aksen (pitch):', 
           "R_y", 
           "\\begin{bmatrix} \\cos(\\theta) & 0 & \\sin(\\theta) \\\\ 0 & 1 & 0 \\\\ -\\sin(\\theta) & 0 & \\cos(\\theta) \\end{bmatrix}",
           useLPrint)
    
    _output('Rotation omkring z-aksen (yaw):', 
           "R_z", 
           "\\begin{bmatrix} \\cos(\\psi) & -\\sin(\\psi) & 0 \\\\ \\sin(\\psi) & \\cos(\\psi) & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}",
           useLPrint)
    
    # Rotation omkring x-aksen (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, m.cos(roll), -m.sin(roll)],
        [0, m.sin(roll),  m.cos(roll)]
    ])

    # Rotation omkring y-aksen (pitch)
    Ry = np.array([
        [m.cos(pitch), 0, m.sin(pitch)],
        [0, 1, 0],
        [-m.sin(pitch), 0, m.cos(pitch)]
    ])

    # Rotation omkring z-aksen (yaw)
    Rz = np.array([
        [m.cos(yaw), -m.sin(yaw), 0],
        [m.sin(yaw),  m.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Sammensæt rotationsmatrice baseret på konvention
    if XYZ == True:
        R = Rx @ Ry @ Rz  # X-Y-Z konvention
        _output('Total rotationsmatrice (X-Y-Z konvention):', "R", "R_x \\cdot R_y \\cdot R_z", useLPrint)
    if XYZ == False:
        R = Rz @ Ry @ Rx  # Z-Y-X konvention (ofte brugt i luftfart)
        _output('Total rotationsmatrice (Z-Y-X konvention):', "R", "R_z \\cdot R_y \\cdot R_x", useLPrint)

    _output('Rotationsmatrice (numerisk):', "R", R, useLPrint)
    return R

def calculate_global_inertia(R, I_local, useLPrint=True):
    """
    Transformerer inertitensoren fra lokal til global koordinatsystem.
    
    Når dronen roterer, ændres inertimomentets repræsentation i det
    globale koordinatsystem. Formlen er: I_global = R @ I_local @ R^T
    
    Parameters:
    -----------
    R : ndarray (3x3)
        Rotationsmatrice
    I_local : ndarray (3x3)
        Lokal inertitensor
    useLPrint : bool
        If True, use formatted Jupyter output; if False, use plain print
    
    Returns:
    --------
    I : ndarray (3x3)
        Inertitensor i globalt koordinatsystem
    """
    _output('Transformation af inertitensor til globalt koordinatsystem:', 
           "I_{global}", 
           "R \\cdot I_{local} \\cdot R^T",
           useLPrint)
    
    # Transponeret rotationsmatrice
    R_t = R.T

    # Transformer til globalt koordinatsystem
    I = R @ I_local @ R_t
    
    _output('Global inertitensor (numerisk):', "I_{global}", I, useLPrint)
    return(I)

def total_force(R, f1, f2, f3, f4, drone_mass, payload_f, useLPrint=True):
    """
    Beregner den totale kraft på dronen i globalt koordinatsystem.
    
    Samler alle kræfter: rotorkræfter (transformeret til globalt),
    tyngdekraft og eventuel payload kraft.
    
    Parameters:
    -----------
    R : ndarray (3x3)
        Rotationsmatrice
    f1, f2, f3, f4 : ndarray (3x1)
        Rotorkræfter i lokal koordinatsystem
    drone_mass : float
        Dronens masse (kg)
    payload_f : ndarray (3x1) eller None
        Payload kraft i globalt koordinatsystem
    useLPrint : bool
        If True, use formatted Jupyter output; if False, use plain print
    
    Returns:
    --------
    f_total : ndarray (3x1)
        Total kraft i globalt koordinatsystem
    """
    _output('Total kraft beregnes ved summering af alle kræfter:', 
           "F_{total}", 
           "R \\cdot (f_1 + f_2 + f_3 + f_4) + F_{gravity} + F_{payload}",
           useLPrint)
    
    # Tyngdekraft i globalt koordinatsystem (peger nedad)
    mg = drone_mass * -9.82  # g = 9.82 m/s²
    gravity = np.array([[0],[0],[mg]])

    # Summer alle rotorkræfter i lokalt koordinatsystem
    f_local = f1 + f2 + f3 + f4
    
    # Transformer rotorkræfter til globalt koordinatsystem
    f_global = R @ f_local

    # Tilføj payload hvis den findes (payload er allerede i globalt koordinatsystem)
    if payload_f is not None:
        f_total = f_global + payload_f + gravity
    else:
        # Kun rotorkræfter og tyngdekraft
        f_total = f_global + gravity
    
    _output('Total kraft (numerisk):', "F_{total}", f_total, useLPrint)
    return f_total
    

    

def total_torque(R, f1, f2, f3, f4, L, useLPrint=True):
    """
    Beregner det totale moment (torque) på dronen.
    
    Momentet beregnes ved krydsproduktet af positionsvektorer og kræfter
    for hver rotor: τ = r × F. Beregnes først i lokalt koordinatsystem,
    derefter transformeret til globalt.
    
    Parameters:
    -----------
    R : ndarray (3x3)
        Rotationsmatrice
    f1, f2, f3, f4 : ndarray (3x1)
        Rotorkræfter i lokal koordinatsystem
    L : float
        Afstand fra massecentrum til rotor (m)
    useLPrint : bool
        If True, use formatted Jupyter output; if False, use plain print
    
    Returns:
    --------
    tau_global : ndarray (3,)
        Totalt moment i globalt koordinatsystem
    """
    _output('Totalt moment beregnes ved krydsproduktet:', 
           "\\tau", 
           "\\sum (r_i \\times F_i)",
           useLPrint)
    
    # Rotorpositioner i lokalt koordinatsystem (jævnt fordelt i + konfiguration)
    # Rotor 1: foran (+x)
    # Rotor 2: højre (+y)
    # Rotor 3: bag (-x)
    # Rotor 4: venstre (-y)
    r1 = np.array([ L, 0, 0])
    r2 = np.array([ 0, L, 0])
    r3 = np.array([-L, 0, 0])
    r4 = np.array([ 0,-L, 0])

    # Beregn moment for hver rotor: τ = r × F (krydsproduktet)
    # Flatten() konverterer fra (3,1) til (3,) for np.cross
    tau1 = np.cross(r1, f1.flatten())
    tau2 = np.cross(r2, f2.flatten())
    tau3 = np.cross(r3, f3.flatten())
    tau4 = np.cross(r4, f4.flatten())

    # Totalt moment i lokalt koordinatsystem
    tau_local = tau1 + tau2 + tau3 + tau4
    
    _output('Moment i lokalt koordinatsystem:', "\\tau_{local}", tau_local, useLPrint)
    
    # Transformer til globalt koordinatsystem
    tau_global = R @ tau_local
    
    _output('Moment i globalt koordinatsystem:', "\\tau_{global}", tau_global, useLPrint)
    return tau_global


def calculate_drone_dynamics(roll, pitch, yaw, f1, f2, f3, f4, drone_mass, L, I_local, 
                            omega=np.array([0.0, 0.0, 0.0]), payload_f=None, 
                            XYZ=True, useLPrint=True):
    """
    Beregner drone dynamik: accelerationer (lineære og vinkler).
    
    Parameters:
    -----------
    roll : float
        Rotation omkring x-aksen (radianer)
    pitch : float
        Rotation omkring y-aksen (radianer)
    yaw : float
        Rotation omkring z-aksen (radianer)
    f1, f2, f3, f4 : ndarray (3x1)
        Rotorkræfter i lokal koordinatsystem [Fx, Fy, Fz]
    drone_mass : float
        Dronens masse (kg)
    L : float
        Afstand fra massecentrum til rotor (m)
    I_local : ndarray (3x3)
        Lokal inertitensor
    omega : ndarray (3,), optional
        Vinkelhastighed [wx, wy, wz] (rad/s), default=[0,0,0]
    payload_f : ndarray (3x1), optional
        Payload kraft i globalt koordinatsystem, default=None
    XYZ : bool, optional
        True for X-Y-Z konvention, False for Z-Y-X, default=True
    useLPrint : bool, optional
        If True, use formatted Jupyter output, default=True
    
    Returns:
    --------
    dict med keys:
        'rotation_matrix': Rotationsmatrice (3x3)
        'global_inertia': Global inertitensor (3x3)
        'total_force': Total kraft (3x1)
        'total_torque': Totalt moment (3,)
        'linear_acceleration': Lineær acceleration (3x1)
        'angular_acceleration': Vinkelacceleration (3,)
    """
    
    # 1. Beregn rotationsmatrice
    R = derive_rotation_matrix(roll, pitch, yaw, XYZ, useLPrint)

    # 2. Beregn global inertitensor
    I_global = calculate_global_inertia(R, I_local, useLPrint)

    # 3. Beregn total kraft
    F_total = total_force(R, f1, f2, f3, f4, drone_mass, payload_f, useLPrint)

    # 4. Beregn totalt moment
    tau_global = total_torque(R, f1, f2, f3, f4, L, useLPrint)

    # ============================================================================
    # LINEÆR ACCELERATION (Newton's 2. lov: F = ma → a = F/m)
    # ============================================================================
    _output("Newtons 2. lov for lineær bevægelse:", 
           "a", 
           "\\frac{F_{total}}{m}",
           useLPrint)
    
    v_dot = F_total / drone_mass
    _output('Lineær acceleration (numerisk):', "a", v_dot, useLPrint)

    # ============================================================================
    # VINKELACCELERATION (Euler's rotationsligning)
    # ============================================================================
    _output("Eulers rotationsligning:", 
           "\\dot{\\omega}", 
           "I^{-1} \\cdot (\\tau - \\omega \\times (I \\cdot \\omega))",
           useLPrint)
    
    # Beregn I·ω (angular momentum)
    Iomega = I_global @ omega

    # Beregn gyroskopisk led (ω × (I·ω))
    gyroscopic = np.cross(omega, Iomega)

    # Beregn vinkelacceleration
    omega_dot = np.linalg.inv(I_global) @ (tau_global - gyroscopic)
    _output('Vinkelacceleration (numerisk):', "\\dot{\\omega}", omega_dot, useLPrint)
    
    return {
        'rotation_matrix': R,
        'global_inertia': I_global,
        'total_force': F_total,
        'total_torque': tau_global,
        'linear_acceleration': v_dot,
        'angular_acceleration': omega_dot
    }

