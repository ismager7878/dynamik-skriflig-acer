import sympy as sp
import numpy as np
def moment_of_inertia_rod_com(m, L):
    """Calculate the moment of inertia of a rectangular rod about its center.

    Parameters:
    m : float
        Mass of the rod.
    L : float
        Length of the rod.
    Returns:
    float
        The moment of inertia of the rod about its center.
    """
    I = (m*L**2)/12
    return I

def s_vec(q, link_length):
        return sp.Matrix([[link_length*sp.cos(q)], [link_length*sp.sin(q)], [0]])

def recursive_speed_and_acc(L, joint_traj, t):
    """
    Generate a recursive trajectory based on the input trajectory and link lengths.

    Parameters:
    L : list of float
        List of link lengths.
    traj : list of sympy expressions
        List of trajectory expressions for each joint.

    Returns:
    list of sympy expressions
        The resulting recursive trajectory.
    """

    n = len(L)
    if len(joint_traj) != n:
        raise ValueError("Length of traj must match length of L")
    
    # Build positions, velocities and accelerations for all links
    angular_velocity = []
    angular_acceleration = []
    body_frame_vecolity = []
    body_frame_acceleration = []
    com_velocity = []
    com_acceleration = []
    
    z = sp.Matrix([0,0,1])

    cumulative_angle = []
    for i in range(n):
        if i == 0:
            cumulative_angle.append(joint_traj[0])
        else:
            cumulative_angle.append(cumulative_angle[i-1] + joint_traj[i])
    
    for i in range(n):
        if (i == 0):
            s_c0 = s_vec(cumulative_angle[0], L[0]/2)
            angular_velocity.append(sp.diff(joint_traj[0], t))
            angular_acceleration.append(sp.diff(angular_velocity[0], t))
            com_velocity.append((angular_velocity[0]*z).cross(s_c0))
            com_acceleration.append(sp.diff(com_velocity[0], t))
            body_frame_vecolity.append(sp.Matrix([[0], [0], [0]]))
            body_frame_acceleration.append(sp.Matrix([[0], [0], [0]]))
            continue

        s_ci = s_vec(cumulative_angle[i], L[i]/2)
        s_li = s_vec(cumulative_angle[i-1], L[i-1])

        angular_velocity.append(angular_velocity[i-1] + sp.diff(joint_traj[i], t))
        angular_acceleration.append(sp.diff(angular_velocity[i], t))

        body_frame_vecolity.append(body_frame_vecolity[i-1] + (angular_velocity[i-1]*z).cross(s_li))
        body_frame_acceleration.append(body_frame_acceleration[i-1] + (angular_acceleration[i-1]*z).cross(s_li) + (angular_velocity[i-1]*z).cross((angular_velocity[i-1]*z).cross(s_li)))

        com_velocity.append(body_frame_vecolity[i] + (angular_velocity[i]*z).cross(s_ci))
        com_acceleration.append(body_frame_acceleration[i] + (angular_acceleration[i]*z).cross(s_ci) + (angular_velocity[i]*z).cross((angular_velocity[i]*z).cross(s_ci)))


    return body_frame_vecolity, body_frame_acceleration, com_velocity, com_acceleration, angular_velocity, angular_acceleration
