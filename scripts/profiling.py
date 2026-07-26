import numpy as np
from pyinstrument import profile

from uniplot import plot

print("Warming up ....")
ys = np.random.random(10)
plot(ys)

ys = np.random.random(1_000_000)

with profile(interval=0.0001):
    plot(ys)

print("Done.")
