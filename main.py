import numpy as np
import matplotlib.pyplot as plt


def ex1_1():
    A = np.linspace(-1.3, 2.5, 64)
    print(A)


def ex1_2(n):
    a = [1, 2, 3]
    b = np.tile(a, n)
    print(b)


def ex1_3():
    a = np.arange(1, 20, 2)
    print(a)


def ex1_4():
    z = np.zeros((10, 10))
    p = np.pad(z, (1,), 'constant', constant_values=(1,))
    print(p)


def ex1_5():
    z = np.zeros((8, 8))
    z[::2, ::2] = 1
    z[1::2, 1::2] = 1
    print(z)


def ex1():
    ex1_1()
    ex1_2(5)
    ex1_3()
    ex1_4()
    ex1_5()


def ex2_2():
    x = np.linspace(-1, 1, 250)
    y = (x ** 2) * np.sin(1 / (x ** 2))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def ex2_3():
    x = np.linspace(5, 25, 250)
    y = np.abs(1 - ((1 / (1 + x ** 2)) / (1 / x ** 2)))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def ex2():
    # ex2_2()
    ex2_3()


def ex3_1():
    x, y = np.meshgrid(np.linspace(-np.pi, np.pi, 250),
                       np.linspace(-np.pi, np.pi, 250))

    f = np.sin(x) * np.sin(y)

    plt.figure()
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-np.pi, np.pi)
    plt.pcolormesh(x, y, f)
    plt.show()


def ex3_2(n):
    a = np.vstack([[i + j for i in range(n)] for j in range(n)])
    print(a)


def ex3():
    # ex3_1()
    ex3_2(10)


def ex4_1():
    x = np.linspace(0, 1, 250)
    fc = np.cos(x)
    fs = np.sin(x)
    a = np.vstack([(c, s) for c, s in zip(fc, fs)])
    print(a)


def ex4_2():
    r = np.random.rand(3, 5)
    print(r)
    print(np.sum(r))
    print(np.sum(r, axis=1))
    print(np.sum(r, axis=0))


def ex4_3():
    r = np.random.rand(5, 5)
    r_c2 = r[:, 1]  # get second column
    src2_i = np.argsort(r_c2)
    print(r)
    print(src2_i)
    print(np.take_along_axis(r, np.tile(src2_i, (5, 1)), axis=1))


def ex4():
    # ex4_1()
    # ex4_2()
    ex4_3()


def main():
    # ex1()
    # ex2()
    # ex3()
    ex4()


if __name__ == '__main__':
    main()
