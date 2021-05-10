import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ITER=10
programs = ('test/paralelo1.txt', 'test/paralelo2.txt', 'test/paralelo3.txt', 'secuencial.txt')
data = []

def getData(filename):

    lista = []
    subLista = []

    with open(filename) as f:
        tmp = f.readline()
        while tmp is not '':
            subLista.append(tmp)
            tmp = f.readline()
        lista.append(tmp_list)
        subLista = []
    return lista

for file in programs:
   data.append(getData(file))

# We have list of Program, and a program consist of N list of strings
# but we know that
for i in data:
    offset = 0
    kernel_avg = 0
    total_avg = 0
    gFlops_avg = 0


    for i in range(0,offset-1):
        kernel_avg = kernel_avg + i[1].split(' ')[2]
