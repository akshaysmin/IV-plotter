import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animateFunc(i):
    print(i)
    x = np.linspace(-10,10,i+1)
    y = x**2
    print(x[0],x[-1])

    ax1.clear()
    ax1.plot(x,y,x,y+2)
    time.sleep(1)
    i += 1

ani = animation.FuncAnimation(fig, animateFunc, interval = 100)#interval= milliseconds for refresh rate

plt.show()

