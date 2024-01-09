import numpy as np

def obj(q, x):
    return x.T@q@x

def greedy_rounding(q):
    n = q.shape[0]
    solution = np.full((n), 0.5)
    val = obj(q, solution)
    for i in range(n):
        solution[i] = 0
        val0 = obj(q, solution)
        solution[i] = 1
        val1 = obj(q, solution)
        if val0 < val1:
            solution[i] = 0
    return solution

def main():
    x = np.array([0, 0, 0])
    q = np.eye(3)
    q[0,0] = -3
    q[1,1] =  2
    q[2,2] = -2

    o = obj(q, x)
    print(o)

    solution = greedy_rounding(q)
    print(solution)

if __name__ == "__main__":
    main()
