import numpy as np
from lib.trilaterate import trilaterate_2d

TRIANGLE_BASE = 1
TRIANGLE_HEIGHT = 0.8660254038 # sqrt(3)/2

VERTICES1 = np.array([
    [-TRIANGLE_BASE/2,TRIANGLE_HEIGHT/3],
    [-74, -292],
    [-9.36, 91.96], # Warsaw (center = Radom)
    [-9.36, 91.96], # Warsaw (center = Radom)
], dtype=np.float32)

VERTICES2 = np.array([
    [TRIANGLE_BASE/2,TRIANGLE_HEIGHT/3],
    [-71.3, -291.4],
    [31.22, -198.03], # Nowy Sącz (center = Radom)
    [-478.7, -146.7] # Prague (center = Radom)
],  dtype=np.float32)

VERTICES3 = np.array([
    [0,TRIANGLE_HEIGHT*2/3],
    [-75.8, -289],
    [-292.87, 111.61], # Poznań (center = Radom)
    [-1341.2, -282.2] # Paris (center = Radom)
], dtype=np.float32)

AMPS1 = np.array([
    [0.65], 
    [3.03],
    [92.4], # Radom to Warsaw
    [543.9] # Berlin to Warsaw
], dtype=np.float32)

AMPS2 = np.array([
    [0.47],
    [4.12],
    [200.6], # Radom to Nowy Sącz
    [281.1], # Berlin to Prague
], dtype=np.float32)

AMPS3 = np.array([
    [0.61],
    [5.78],
    [310.3], # Radom to Poznań
    [886.8] # Berlin to Paris
], dtype=np.float32)


def main():
    points = trilaterate_2d((VERTICES1, VERTICES2, VERTICES3), (AMPS1, AMPS2, AMPS3))
    print(points)


if __name__ == '__main__':
    main()