from cv2 import cvtColor, COLOR_RGB2GRAY, resize
from math import modf

# Hu's Invariant Moments 
def M(p, q, f):
    result = 0
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            result += x**p * y**q * f[x][y]
    return result


def μ(p, q, x_hat, y_hat, f):
    result = 0
    for x in range(f.shape[0]):
        for y in range(f.shape[1]):
            result += (x - x_hat)**p * (y - y_hat)**q * f[x,y]
    return result


def γ(p, q):
    return (p + q) / 2 + 1


def η(p, q, xhat, yhat, f):
    return μ(p, q, xhat, yhat, f) / μ(0, 0, xhat, yhat, f) ** γ(p, q)


def calculate_invariants(image):
    image = cvtColor(image, COLOR_RGB2GRAY)
    if image.shape[0] > 32 or image.shape[1] > 32:
        image = resize(image, (32, 32))

    M10 = M(1, 0, image)
    M01 = M(0, 1, image)
    M00 = M(0, 0, image)

    xhat = M10 / M00
    yhat = M01 / M00

    η20 = η(2, 0, xhat, yhat, image)
    η02 = η(0, 2, xhat, yhat, image)
    η11 = η(1, 1, xhat, yhat, image)
    η12 = η(1, 2, xhat, yhat, image)
    η21 = η(2, 1, xhat, yhat, image)
    η30 = η(3, 0, xhat, yhat, image)
    η03 = η(0, 3, xhat, yhat, image)

    Φ1 = η20 + η02
    Φ2 = (η20 - η02)**2 + 4*η11**2
    Φ3 = (η30 - 3*η12)**2 + (3*η21-η03)**2
    Φ4 = (η30 + η12)**2 + (η21 + η03)**2
    Φ5 = (η30 - 3*η12) * (η30 + η12) * ((η30 + η12)**2 - 3*(η21 + η03)**2) + (3*η21 - η03) * (η21+η03) * (3*(η30 + η12)**2 - (η21 + η03)**2)
    Φ6 = (η20 - η02) * ((η30 + η12)**2 - (η21 + η03)**2) + 4*η11*(η30+η12) * (η21+η03)
    Φ7 = (3*η21 - η03) * (η30 + η12) * ((η30 + η12)**2 - 3*(η21 + η03)**2) - (η30 - 3*η12) * (η21 + η03) * (3*(η30 + η12)**2 - (η21 + η03)**2)

    return (Φ1, Φ2, Φ3, Φ4, Φ5, Φ6, Φ7)


def calculate_first_invariant(image):
    image = cvtColor(image, COLOR_RGB2GRAY)
    if image.shape[0] > 32 or image.shape[1] > 32:
        image = resize(image, (32, 32))

    M10 = M(1, 0, image)
    M01 = M(0, 1, image)
    M00 = M(0, 0, image)

    xhat = M10 / M00
    yhat = M01 / M00

    η20 = η(2, 0, xhat, yhat, image)
    η02 = η(0, 2, xhat, yhat, image)

    Φ1 = η20 + η02

    return Φ1

#TODO: calculate hash string from invariant
def invariant_hash(image):
    invariant = calculate_first_invariant(image)
    return  f"{modf(abs(invariant))[0]:.24f}"[2:]