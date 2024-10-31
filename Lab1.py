import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse import csr_array, eye_array
from scipy.sparse.linalg import spsolve

matplotlib.use('TkAgg')


def get_polynomial(x, n):
    if n == 0:
        return 1
    if n == 1:
        return x
    return (2*(n - 1) + 1) / n * x * get_polynomial(x, n - 1) - (n - 1) / n * get_polynomial(x, n - 2)

n = 5000
k = 6
a = -1
b = 1
x_left_borders, h = np.linspace(a, b, n, endpoint=False, retstep=True)
x_centers = x_left_borders + h / 2
x_right_borders = x_centers + h / 2

polynomials = []

for i in range(k):
    temp = []
    for j in range(n):
        temp.append(get_polynomial(x_centers[j], i))
    polynomials.append(temp)


polynomials = np.array(polynomials)


data_A = 1 / (h ** 2) * np.concatenate((x_left_borders[1:-1] ** 2 - 1, np.array([(1 - h) ** 2 - 1])))
data_B = 1 / (h ** 2) * np.concatenate((np.array([1 - (1 - h) ** 2]), 2 - x_left_borders[1:-1] ** 2 - x_right_borders[1:-1] ** 2, np.array([1 - (1 - h)**2])))
data_C = 1 / (h ** 2) * np.concatenate((np.array([(1 - h) ** 2 - 1]), x_right_borders[1:-1] ** 2 - 1))

i_ind_main_diag = np.arange(n)
j_ind_main_diag = np.arange(n)

i_ind_bot_diag = np.arange(1, n)
j_ind_bot_diag = np.arange(n - 1)

i_ind_top_diag = np.arange(n - 1)
j_ind_top_diag = np.arange(1, n)

i_ind = np.concatenate((i_ind_main_diag, i_ind_bot_diag, i_ind_top_diag))
j_ind = np.concatenate((j_ind_main_diag, j_ind_bot_diag, j_ind_top_diag))

data = np.concatenate((data_B, data_A, data_C))

L = csr_array((data, (i_ind, j_ind)), shape=(n, n))

true_lambdas = np.array([i * (i + 1) for i in range(k)])
lambdas = true_lambdas - 0.01
eigen_vectors = []

for i in range(len(lambdas)):
    v_old = np.random.rand(n)
    v_old_n = v_old / np.linalg.norm(v_old)
    eye_matrix = eye_array(n)
    
    cnt = 1
    prev_lambda = lambdas[i]
    eps = 1e-8
    
    
    while True:
        v_new = spsolve(L - lambdas[i] * eye_matrix, v_old_n)
        v_new_n = v_new / np.linalg.norm(v_new)
        
        prev_lambda = lambdas[i]
        lambdas[i] = np.dot(L @ v_new_n, v_new_n) / np.dot(v_new_n, v_new_n)

        if np.abs(lambdas[i] - prev_lambda) < eps:
            eigen_vectors.append(v_new_n)
            break
        
        v_old_n = v_new_n
        cnt += 1
        
        if cnt % 100 == 0:
            print(f"Итерация {cnt}: λ[{i}] = {lambdas[i]}")

    print(f"Собственное значение {i+1}: {lambdas[i]}")

fig, ax = plt.subplots(2, 3, figsize=(10, 15))
polynomials
for i in range(k):
    polynomials_n = polynomials[i] / np.linalg.norm(polynomials[i])
    ax[i // 3, i % 3].scatter(x_centers, eigen_vectors[i], label=f'Собственная функция {i + 1}', color='blue')
    ax[i // 3, i % 3].plot(x_centers, polynomials_n, label=f"Полином {i+1}", color='red', linestyle='--')
    ax[i // 3, i % 3].set_title(f"Собственная функция {i+1}")
    ax[i // 3, i % 3].legend()

plt.savefig("Lab1.png")

fig, ax = plt.subplots()
ax.plot(x_centers, eigen_vectors[0] * 40 + lambdas[0], label="Полином 1", color='red', linestyle='--', ms=1)
ax.plot(x_centers, eigen_vectors[1] * 40 + lambdas[1], label="Полином 2", color='green', linestyle='--', ms=1)
ax.plot(x_centers, eigen_vectors[2] * 40 + lambdas[2], label="Полином 3", color='blue', linestyle='--', ms=1)
ax.plot(x_centers, eigen_vectors[3] * 40 + lambdas[3], label="Полином 4", color='orange', linestyle='--', ms=1)
ax.plot(x_centers, eigen_vectors[4] * 40 + lambdas[4], label="Полином 5", color='purple', linestyle='--', ms=1)
ax.plot(x_centers, eigen_vectors[5] * 40 + lambdas[5], label="Полином 6", color='black', linestyle='--', ms=1)
plt.savefig('Eigen_vectors.png')
plt.show()


