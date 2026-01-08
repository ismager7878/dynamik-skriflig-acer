from sympy import latex
from IPython.display import Markdown, display

def lPrint(label, res, expr, sp=True):
    display(Markdown(f"**{label}**: $${res}={latex(expr) if sp else expr}$$"))