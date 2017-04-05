
all:
	nvcc -Wno-deprecated-gpu-targets -DDIM=3 -std=c++11 trilateration.cu -o 3dtri
	nvcc -Wno-deprecated-gpu-targets -DDIM=2 -std=c++11 trilateration.cu -o 2dtri

dbg:
	nvcc -g -Wno-deprecated-gpu-targets -DRSEL_DEBUG -DDIM=3 -std=c++11 trilateration.cu -o 3dtri
	nvcc -g -Wno-deprecated-gpu-targets -DRSEL_DEBUG -DDIM=2 -std=c++11 trilateration.cu -o 3dtri

.PHONY: clean

clean:
	rm -f *.out 2dtri 3dtri 2>/dev/null

