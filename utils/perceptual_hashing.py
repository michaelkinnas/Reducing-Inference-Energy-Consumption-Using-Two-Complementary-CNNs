from cv2 import cvtColor, COLOR_RGB2GRAY, resize
from numpy import asmatrix, zeros, shape, linspace, array, power, round, append, absolute, meshgrid, sign
from math import pi
from scipy.special import comb
from math import modf


def __calculateMoments(data_matrix, order):
    """
    It calculates geometric moments.
    
    :param data_matrix: numpy array
    :return: geometric moments: numpy array
    """
    M = zeros((order + 1, order + 1))
    (n1, n2) = shape(data_matrix)
    m00 = data_matrix.sum()
    w = linspace(1, n2, n2)
    v = linspace(1, n1, n1)
    if m00 != 0:
        tx = ((data_matrix * array([w]).T).sum()) / float(m00)
        ty = ((data_matrix.T * array([v]).T).sum()) / float(m00)
    else:
        tx = 0
        ty = 0
    a = w - tx
    c = v - ty
    for i in range(1, order + 1 + 1):
        for j in range(1, order + 2 - i + 1):
            p = i - 1
            q = j - 1
            A = power(a, p)
            C = power(c, q)
            oo = C * data_matrix * (array([A]).T)
            M[i - 1, j - 1] = oo
    if order > 0:
        M[0, 1] = 0
        M[1, 0] = 0
    return M


def __geometricMomentsToComplexMoments(gm, order):
    """
    It uses the geometric momements to calculate the complex moments.
    
    :param gm: geometric moments: numpy array
    :return: complex moments: numpy array
    """
    c = zeros((order + 1, order + 1)).astype(complex)
    for p in range(0, order + 1):
        for q in range(0, order - p + 1):
            for k in range(0, p + 1):
                pk = comb(p, k)
                for w in range(0, q + 1):
                    qw = comb(q, w)
                    c[p, q] = c[p, q] + pk * qw * (-1) ** (q - w) * 1j ** (p + q - k - w) * gm.item(
                        (k + w, p + q - k - w))
    return c


def calculate_complex_invariants(image):
    """
    It calculates the invariants.
    
    :param data_matrix: numpy array
    """
    if image.shape[2] == 3:
        image = cvtColor(image, COLOR_RGB2GRAY)
    if image.shape[0] > 32 or image.shape[1] > 32:
        image = resize(image, (32, 32))
    data_matrix = asmatrix(image)

    invariants = array([])
    order = 3

    p0 = 1
    q0 = 2
    if order == 2:
        p0 = 0
        q0 = 2
    if p0 > q0:
        aux = p0
        p0 = q0
        q0 = aux
    m = __calculateMoments(data_matrix, order)
    c = __geometricMomentsToComplexMoments(m, order)
    c = round(c, 5)
    tmpx = linspace(0, order, order + 1)
    tmpy = linspace(0, order, order + 1)

    qm, pm = meshgrid(tmpx, tmpy)
    c = c / ((m[0, 0] ** ((qm + pm + 2) / 2.0)) * 1.0)
    c = c * ((pm + qm) / 2 + 1) * pi ** ((pm + qm) / 2.0)
    ident = q0 - p0
    ni = 0
    pwi = array([])
    if ident == 0:
        for r1 in range(2, order + 1, 2):
            p = round(r1 / 2.0, 0)
            tmp = c[p, p]
            invariants = append(invariants, tmp.real)
            pwi = append(pwi, 1)
            ni = ni + 1
    else:
        for r1 in range(max(2, ident), order + 1, ident):
            for p in range(int(round(float(r1) / 2.0, 0)), r1 + 1):
                q = r1 - p
                if (p - q) % ident == 0:
                    tmp = c[p, q] * c[p0, q0] ** ((p - q) / ident)
                    invariants = append(invariants, tmp.real)
                    pwi = append(pwi, (1 + (p - q) / ident))
                    ni = ni + 1
                    if (p > q) and (p != q0 or q != p0):
                        tmp2 = c[p, q] * c[p0, q0] ** ((p - q) / ident)
                        invariants = append(invariants, tmp2.imag)
                        pwi = append(pwi, 1 + (p - q) / ident)
                        ni = ni + 1
    invariants = sign(invariants) * (absolute(invariants) ** (1 / pwi))
    return invariants


def complex_invariants_hash_string(image):
    invariants = calculate_complex_invariants(image)
    x, _ = modf(invariants[0])
    return f"{x:.24f}"[2:]


def complex_invariants_hash_addition_float(image):
    invariants = calculate_complex_invariants(image)
    return invariants[0] + invariants[1] + invariants[3] + invariants[4]


def dhash(image, hash_size = 8):
    image = cvtColor(image, COLOR_RGB2GRAY)
    resized = resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])