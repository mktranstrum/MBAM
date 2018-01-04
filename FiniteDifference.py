# Finite difference formulas for higher accuracy

# Forward Difference Formauls

def FD(f, x, v, h):
    return (f(x + h*v) - f(x))/h


def FD2(f, x, v, h):
    return (-f(x + 2*h*v) + 4*f(x + h*v) - 3*f(x))/(2*h)


def FD3(f, x, v, h):
    return (f(x + 4*h*v) - 12*f(x + 2*h*v) + 32*f(x + h*v) - 21*f(x))/(12*h)


def FD4(f, x, v, h):
    return (-f(x + 8*h*v) + 28*f(x + 4*h*v) - 224*f(x + 2*h*v) + 512*f(x + h*v) - 315*f(x))/(168*h)


# Center Difference Formauls
def CD(f, x, v, h):
    return (f(x + h*v) - f(x-h*v))/(2*h)


def CD4(f, x, v, h):
    return (-f(x + 2*h*v) + 8*f(x + h*v) - 8*f(x - h*v) + f(x - 2*h*v))/(12*h)


# Avv Formulas

def AvvCD(f, x, v, h):
    return ( f(x + h*v) + f(x - h*v) - 2*f(x))/(h**2)


def AvvCD4(f, x, v, h):
    return ( -f(x + 2*h*v) + 16*f(x + h*v) + 16*f(x - h*v) - f(x - 2*h*v) - 30*f(x))/(12*h**2)


# Auv formulas

def AuvCD(f, x, u, v, h):
    return ( f(x + h*u + h*v) - f(x + h*u - h*v) - f(x - h*u + h*v) + f(x - h*u - h*v))/(4*h**2)


def AuvCD4(f, x, u, v, h):
    return ( -f(x + 2*h*u + 2*h*v) + f(x + 2*h*u - 2*h*v) + 16*f(x + h*u + h*v) - 16*f(x + h*u - h*v) - 16*f(x - h*u + h*v) + 16*f(x - h*u - h*v) + f(x - 2*h*u + 2*h*v) - f(x - 2*h*u - 2*h*v))/(48*h**2)

