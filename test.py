import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib
from scipy.sparse import lil_matrix

matplotlib.use('TkAgg')


h = float(input())
c = float(input())
a = 2
tau = c * h / a
epsilon = 1
xlimit = 5
t = np.arange(0, 3 + tau, tau)
x = np.arange(0, xlimit + h, h)
x0 = 2
fig, ax = plt.subplots(2, 1, figsize=(15, 20))
ax[0].set_xlim(0, 5)
ax[0].set_ylim(0, 1.2)
ax[1].set_xlim(0, 3)
ax[1].set_ylim(0, 0.8)


line1, = ax[0].plot([], [], lw=3, label='Приблизительное решение')
line2, = ax[0].plot([], [], lw=3, label='Точное решение')
line3, = ax[1].plot([], [], lw=3, label='Ошибка')
ax[0].set_title('Приблизительное и точное решения')
ax[1].set_title('Зависимость ошибки от времени')



def xi(x, x0, epsilon):
    return np.abs(x - x0) / epsilon

def phi1(x):
    return np.heaviside(1 - xi(x, x0=x0, epsilon=epsilon), 1)

def phi2(x):
    return phi1(x) * (1 - np.power(xi(x, x0=x0, epsilon=epsilon), 2))

def phi3(x):
    return phi1(x) * np.exp(-np.power(xi(x, x0=x0, epsilon=epsilon), 2) / np.abs(1 - np.power(xi(x, x0=x0, epsilon=epsilon), 2)))

def phi4(x):
    return phi1(x) * np.power(np.cos(np.pi * xi(x, x0=x0, epsilon=epsilon) / 2), 3)

def real(x, t):
    return phi4((x - a * t) % xlimit)


u_prev = phi4(x)
u_next = np.zeros(len(x))
max_error = np.array([])


def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3,

def update(i):
    global u_prev, u_next, max_error
    if i == 0:
        line1.set_data(x, u_prev)
        line2.set_data(x, real(x, 0))
        max_error = np.append(max_error, np.max(np.abs(u_prev - real(x, 0))))
        line3.set_data(t[:len(max_error)], max_error)
        return line1, line2, line3,
    u_next[1:] = u_prev[1:] - a * tau * (u_prev[1:] - u_prev[:-1]) / h
    
    u_next[0] = u_next[-1]
    u_prev = np.copy(u_next)
    u_next = np.zeros(len(x))
    line1.set_data(x, u_prev)
    line2.set_data(x, real(x, i))
    max_error = np.append(max_error, np.max(np.abs(u_prev - real(x, i))))
    line3.set_data(t[:len(max_error)], max_error[:len(max_error)])
    return line1, line2, line3,

anim = FuncAnimation(fig, update, init_func=init, frames=t, interval=1, blit=True, repeat=False)

ax[0].legend()
ax[1].legend()
plt.show()