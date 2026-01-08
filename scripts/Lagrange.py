import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import pprint

from scripts.l_print import lPrint

t = sp.symbols('t')

def find_forces_lagrange(type: str, trajectories: list, velocities: list, link_lengths: list, masses: list, heights: list, useLPrint=True):
    """
    Find forces/torques using the Lagrange method for multi-link mechanisms.

    **Parameters:**
    - **type** (str): Joint type string, e.g., 'RRR' (revolute) or 'RPR' (prismatic-revolute-prismatic)
    - **trajectories** (list<function>): Functions for each joint variable
    - **velocities** (list): Nx2 array [[ang_vel, trans_vel], ...] for each joint
        - Use 0 for no angular velocity
        - Use sp.Matrix([0,0,0]) for no translational velocity
    - **link_lengths** (list<float>): Length of each link
    - **masses** (list<float>): Mass of each link
    - **heights** (list<float>): Height (y-coordinate) of each link's center of mass
    - **useLPrint** (bool): If True, use formatted Jupyter output; if False, use plain print

    **Returns:**
    - list: Forces/torques for each joint from Euler-Lagrange equations
    """

    if(len(trajectories) != len(type) or len(velocities) != len(type) or len(link_lengths) != len(type) or len(masses) != len(type) or len(heights) != len(type)):
        return "Error: Length of input lists must match length of type string, and all be the same length"

    #Get length of type string
    axisCount = len(type)

    Ts = []  # Kinetic energies
    Vs = []  # Potential energies

    # Calculate T and V for each link
    for i in range(axisCount):
        if(i == 0 and type[i] == 'R'):  # First link with revolute joint
            _output('Kinetic Energy of link 1 is found by fixed axis rotation:', 
                   "T_{rot}", "\\frac{1}{2} I_0 \\omega^2", useLPrint, is_sympy=False)
            
            iO = inertia(masses[i], link_lengths[i], com_dist=link_lengths[i]/2)
            T = kin_energy(masses[i], ang_vel=velocities[i][0], inertia=iO)
            V = potential_energy(masses[i], heights[i])
    
        else:
            Ig = inertia(masses[i], link_lengths[i])

            if(velocities[i][0] == 0 and i == 0):
                _output(f'Kinetic Energy of link {i+1} found by pure translation:', 
                       "T_{trans}", "\\frac{1}{2} m v^2", useLPrint, is_sympy=False)
            else:
                _output(f'Kinetic Energy of link {i+1} found by General motion:', 
                       "T_{general}", "\\frac{1}{2} I \\omega^2 + \\frac{1}{2} m v^2", useLPrint, is_sympy=False)

            T = kin_energy(masses[i], ang_vel=velocities[i][0], trans_vel=velocities[i][1], inertia=Ig)
            V = potential_energy(masses[i], heights[i])
        
        _output('The potential energy is found by:', "V", "m g h", useLPrint, is_sympy=False)

        Ts.append(T)
        Vs.append(V)

        T = T.factor().nsimplify()
        V = V.factor().nsimplify()

        _output(f'Translational Kinetic Energy T{i+1}:', f"T_{i+1}", T, useLPrint)
        _output(f'Potential Energy V{i+1}:', f"V_{i+1}", V, useLPrint)
    
    # Construct Lagrangian: L = T - V
    L = 0
    for i in range(axisCount):
        L += Ts[i] - Vs[i]

    L = L.factor().nsimplify()
    _output('Lagrange Function L:', "L", L, useLPrint)

    # Apply Euler-Lagrange equation for each trajectory
    forceList = []

    for i in range(axisCount):
        # d/dt(∂L/∂q̇) - ∂L/∂q = F
        force = sp.diff(sp.diff(L, sp.diff(trajectories[i], t)), t) - sp.diff(L, trajectories[i])
        force = force.factor().nsimplify()
        forceList.append(force)
        if(type[i] == 'R'):
            label = f'Torque τ{i+1}'
            var_name = f"\\tau_{i+1}"
        else:
            label = f'Force F{i+1}'
            var_name = f"F_{i+1}"
        
        _output(label, var_name, force, useLPrint)

    return forceList


def inertia(mass, link_length, type = "rod", com_dist = 0):
    """Calculate moment of inertia (parallel axis theorem if com_dist given)."""
    if(type == "rod"):
        i = (mass*link_length**2)/12  # I_G for uniform rod
        if(com_dist == 0):
            return i
        
        return i + mass*com_dist**2  # Parallel axis: I_O = I_G + md²

def kin_energy(mass, ang_vel=0, trans_vel=sp.Matrix([0,0,0]), inertia=0):
    """Calculate kinetic energy: pure translation, rotation, or general motion."""
    if(ang_vel == 0):  # Pure translation
        return 1/2 * mass * trans_vel.dot(trans_vel)
    elif(trans_vel == sp.Matrix([0,0,0])):  # Pure rotation
        return 1/2 * inertia * ang_vel**2
    else:  # General motion
        return 1/2 * mass * trans_vel.dot(trans_vel) + 1/2 *  inertia * ang_vel**2
    
def potential_energy(mass, height, g=sp.symbols('g')):
    """Calculate gravitational potential energy: V = mgh."""
    return mass * g * height

def _output(label, var_name, expr, use_lprint=True, is_sympy=True):
    """Helper to handle both lPrint and standard print output."""
    # Substitute derivatives with readable symbols if expr is a sympy expression
    if is_sympy and hasattr(expr, 'subs'):
        expr = _substitute_derivatives(expr)
    
    if use_lprint:
        lPrint(label, var_name, expr, sp=is_sympy)  # Jupyter markdown output with bold
    else:
        print(f'{label}:')  # Plain text output
        if is_sympy:
            pprint(expr)
        else:
            print(f'{var_name} = {expr}')

def _substitute_derivatives(expr):
    """Substitute derivatives with readable velocity and acceleration symbols."""
    if not hasattr(expr, 'atoms'):
        return expr
    
    # Find all derivatives in the expression
    derivatives = expr.atoms(sp.Derivative)
    
    # Create substitution dictionary
    subs_dict = {}
    
    for deriv in derivatives:
        # Get the function being differentiated
        func = deriv.expr
        
        # Check if it's a function of t
        if not hasattr(func, 'func'):
            continue
            
        func_name = str(func.func).replace('(t)', '')
        
        # Count the order of differentiation with respect to t
        deriv_order = sum(1 for arg in deriv.variables if arg == t)
        
        # Determine the appropriate symbol name
        if deriv_order == 1:  # First derivative (velocity/angular velocity)
            # Check if the variable name contains "theta" or similar rotational indicators
            if 'theta' in func_name.lower() or 'phi' in func_name.lower() or 'psi' in func_name.lower():
                # Extract number from function name if present
                num = ''.join(filter(str.isdigit, func_name))
                symbol_name = f'omega{num}' if num else 'omega'
            else:
                # Linear velocity
                num = ''.join(filter(str.isdigit, func_name))
                symbol_name = f'v{num}' if num else 'v'
        elif deriv_order == 2:  # Second derivative (acceleration/angular acceleration)
            # Check if the variable name contains "theta" or similar rotational indicators
            if 'theta' in func_name.lower() or 'phi' in func_name.lower() or 'psi' in func_name.lower():
                # Extract number from function name if present
                num = ''.join(filter(str.isdigit, func_name))
                symbol_name = f'alpha{num}' if num else 'alpha'
            else:
                # Linear acceleration
                num = ''.join(filter(str.isdigit, func_name))
                symbol_name = f'a{num}' if num else 'a'
        else:
            continue  # Skip higher order derivatives
        
        subs_dict[deriv] = sp.symbols(symbol_name)
    
    # Apply substitutions
    return expr.subs(subs_dict)