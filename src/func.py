def func(x: str):
    x = x.lower()  # rende tutto minuscolo
    return x == x[::-1]