from sympy import latex
from IPython.display import Markdown, display
import numpy as np

def lPrint(label, res, expr, sp=True):
    """
    Print formatted output in Jupyter notebooks.
    
    Parameters:
    -----------
    label : str
        Description label
    res : str
        Variable name (LaTeX format)
    expr : sympy expression, numpy array, or str
        Expression to display
    sp : bool
        If True, use sympy latex conversion. If False, format as string/LaTeX
    """
    if sp:
        # Sympy expression - use latex conversion
        formatted_expr = latex(expr)
    else:
        # For numpy arrays, format them nicely
        if isinstance(expr, np.ndarray):
            if expr.ndim == 1:
                # 1D array - format as row vector
                formatted_expr = "\\begin{bmatrix}" + " & ".join([f"{x:.4f}" for x in expr]) + "\\end{bmatrix}"
            elif expr.ndim == 2:
                # 2D array - format as matrix
                rows = []
                for row in expr:
                    if row.shape[0] == 1:
                        rows.append(f"{row[0]:.4f}")
                    else:
                        rows.append(" & ".join([f"{x:.4f}" for x in row]))
                formatted_expr = "\\begin{bmatrix}" + " \\\\ ".join(rows) + "\\end{bmatrix}"
            else:
                formatted_expr = str(expr)
        else:
            # String or other type - use as is
            formatted_expr = expr
    
    display(Markdown(f"**{label}**: $${res}={formatted_expr}$$"))