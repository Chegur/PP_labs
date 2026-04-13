import numpy as np
import sys

def read_matrix(filename):
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        matrix = []
        for line in f:
            row = [float(x) for x in line.split()]
            matrix.append(row)
        return np.array(matrix), n

def verify(result_file, a_file, b_file):
    try:
        A, nA = read_matrix(a_file)
        B, nB = read_matrix(b_file)
        C_cpp, nC = read_matrix(result_file)

        if nA != nB or nA != nC:
            print ("Error size")
            return False
        C_ref = np.dot(A,B)

        if np.allclose(C_cpp, C_ref, rtol=1e-5, atol=1e-8):
            print("Verify complete succesful")
            return True
        else:
            print("Verify Error")
            diff = np.abs(C_cpp - C_ref).max()
            print (f"Max difference: {diff}")
            return False
    except Exception as e:
        print(f"Error {e}")
        return False

def main():
    verify("matrix_c.txt", "matrix_a.txt", "matrix_b.txt")

if __name__ == "__main__":
    main()