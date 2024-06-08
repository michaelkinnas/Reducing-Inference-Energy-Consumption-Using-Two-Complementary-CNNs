from cv2 import cvtColor, COLOR_RGB2GRAY, resize

def dhash(image, hash_size = 8):
    image = cvtColor(image, COLOR_RGB2GRAY)
    resized = resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])