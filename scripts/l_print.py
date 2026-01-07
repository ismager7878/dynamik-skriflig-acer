from sympy import latex
from IPython.display import Markdown, display

def lPrint(label, expr):
    display(Markdown(f"**{label}**: $${latex(expr)}$$"))