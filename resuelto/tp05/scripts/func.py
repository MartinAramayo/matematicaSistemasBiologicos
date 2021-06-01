def sech_sir(x, a, b, c):
    y = a * ( ( np.cosh( b * x + c ) ) ** (-2) )
    return y
