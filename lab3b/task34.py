import numpy as np
import random
import matplotlib.pyplot as plt
from hopfield_net import HopfieldNet
from pict_memory import display_image

def read_data():

    pict_shape = (32, 32)
    pict_size = pict_shape[0] * pict_shape[1]
    test_n = 3

    # import data from .dat file
    data = np.loadtxt("data/pict.dat", delimiter=",", dtype=int).reshape(11, pict_size)[:4]
    train_data = data[:3]

    return train_data

def add_noise(x, p):

    x_d = x.copy()
    len_x = len(x_d)
    n_flip = int(len_x*(p/100))
    indices = list(range(len_x))
    random.shuffle(indices)

    for i in range(n_flip):
        i_flip = indices[i]
        x_d[i_flip] *= -1

    return x_d

def main():

    hopfield = HopfieldNet(1024)
    output_ls = {}
    data = read_data()
    hopfield.train(data)
    print('training done')
    data_c = data

    max_p_ls = []
    x_d_ls = []

    hamming_1 = sum(bit1 != bit2 for bit1, bit2 in zip(data_c[0], data_c[1]))
    hamming_1 += sum(bit1 != bit2 for bit1, bit2 in zip(data_c[0], data_c[2]))
    hamming_2 = sum(bit1 != bit2 for bit1, bit2 in zip(data_c[0], data_c[1]))
    hamming_2 += sum(bit1 != bit2 for bit1, bit2 in zip(data_c[1], data_c[2]))
    hamming_3 = sum(bit1 != bit2 for bit1, bit2 in zip(data_c[2], data_c[1]))
    hamming_3 += sum(bit1 != bit2 for bit1, bit2 in zip(data_c[0], data_c[2]))

    for m in range(2):
        max_p = np.zeros(3)
        for p in np.linspace(0, 100, 10):
            
            output_ls[p] = []
            output = []
            output_print = []
            x_d_ls.append([])
            for i in range(len(data_c)):
                x_run = add_noise(data_c[i], p)
                x_d_ls[-1].append(x_run)
                
                out = hopfield.run(x_run, n_cycles=100)
                count = 0
                output_print.append(out)
                for k, val in enumerate(out):
                    if val == data_c[i][k]:
                        count += 1
                if np.array_equal(out, data_c[i]):
                    output.append(out)
                    max_p[i] = p

                output_ls[p].append(count)

            print("Number of attractors:", len(output))
            print("Attractors:")
            for attractor in output:
                if np.array_equal(attractor, data[0]):
                    print(f"{attractor} (= x1d)")
                elif np.array_equal(attractor, data[1]):
                    print(f"{attractor} (= x2d)")
                elif np.array_equal(attractor, data[2]):
                    print(f"{attractor} (= x3d)")
                elif np.array_equal(attractor, -data[0]):
                    print(f"{attractor} (= -x1d)")
                elif np.array_equal(attractor, -data[1]):
                    print(f"{attractor} (= -x2d)")
                elif np.array_equal(attractor, -data[2]):
                    print(f"{attractor} (= -x3d)")
                else:
                    print(attractor)
            for out in output_print:
                display_image(out, (32,32), 'Distorted pattern:'+str(round(p/100,1)))
                plt.show()
            print('-------')
        max_p_ls.append(max_p)

    max_counts = np.zeros([10,3])
    diff_ls = np.zeros([len(x_d_ls),3])
    for j,count in enumerate(max_p_ls):
        for i in range(3):
            max_counts[j][i] = count[i]

    for j, pats in enumerate(x_d_ls):
        for pat in pats:
            for k, val in enumerate(pat):
                if val == -1:
                    pat[k] = 0
        for i in range(3):
            for i2 in range(3):
                if i != i2:
                    diff_ls[j][i] += np.sum(abs(pats[i]-pats[i2]))
    print(max_counts)

main()


    
#[1-1 1-1 1-1-1 1]
#[-1-1 1-1 1-1-1 1]