from matplotlib import pyplot as plot
import numpy as np
import math

x = np.linspace(0.01, 20, 200, endpoint=True)
K = 10
A = 40

y = K / x + x / 2

plot.plot(x, y)
plot.savefig('plot.jpg')
