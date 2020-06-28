from functools import partial

def f(a,b,c):
    return a+b+c


g = partial(f, b=3, c=2)

print(g(1))