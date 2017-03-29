import matplotlib.pyplot as plt
import subprocess

# TODO: All the cuda info stuff
# U and V are dummy values for now
# Will use something (pycuda or deviceQuery) to properly get these values
U = 1024
V = 64

NUM = [12, 13, 14, 15]

# May have gotten compiler flags wrong here :/
prog = ['./trilateration NUMS=2 ']

configs = [(U,V), (2*U,V), (U,2*V), (2*U,2*V), (U/2,V), (U,V/2), (U,V/2), (U/2,V/2)]

configs = [[n] + list(c) for n in NUM for c in configs]
args = [prog + c for n in NUM for c in configs]

results = [int(subprocess.check_output(arg)) for arg in args]

plt.hist(results)
plt.title('2D trilateration with different GPU configurations')
plt.xlabel('GPU configurations')
plt.ylabel('Errors')
plt.show()
