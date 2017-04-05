#!/usr/bin/env python3

import time
import matplotlib.pyplot as plt
import subprocess
import sys

def timeGPU(params):
    start = time.time()
    for _ in range(1,10): subprocess.call(params)
    return time.time() - start

U = 6   # The number of streaming multiprocessors
V = 192 # The number of CUDA cores per SM

NUM = [12, 13, 14, 15]

# Make the correct executables
if subprocess.call(['make']):
    print(">> ERROR: Failed to make 3d executable.")
    sys.exit(1)

progs = ['./3dtri', './2dtri']

configs = [(U,V), (2*U,V), (U,2*V), (2*U,2*V), (int(U/2),V), (U,int(V/2)), (int(U/2),int(V/2))]

labels = [['U','V'], ['2*U','V'], ['U','2*V'], ['2*U','2*V'], ['U/2','V'], ['U','V/2'], ['U/2','V/2']]
labels = [','.join([str(n)] + l) for n in NUM for l in labels]

configs = [[n] + list(c) for n in NUM for c in configs]

args3d = [[progs[0]] + list(map(str, c)) for c in configs]
args2d = [[progs[1]] + list(map(str, c)) for c in configs]

results3d = [timeGPU(arg) for arg in args3d]
results2d = [timeGPU(arg) for arg in args2d]

plt.figure(1)
plt.bar(range(len(results3d)), results3d, align='center')
plt.xticks(range(len(labels)), labels, size='small', rotation=70)
plt.title('3D trilateration - 10 runs each')
plt.xlabel('GPU configurations')
plt.ylabel('Run Time (s)')

plt.tight_layout()
plt.savefig('results-3d.png')

plt.figure(2)
plt.bar(range(len(results2d)), results2d, align='center')
plt.xticks(range(len(labels)), labels, size='small', rotation=70)
plt.title('2D trilateration - 10 runs each')
plt.xlabel('GPU configurations')
plt.ylabel('Run Time (s)')

plt.tight_layout()
plt.savefig('results-2d.png')
