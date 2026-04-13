import numpy as np
import sys 

def generate_matrix (n, outfilename):
    """Генерирует случайную матрицу n*n"""
    matrix = np.random.rand(n,n)
    with open (outfilename, 'w') as f:
        f.write(f"{n}\n")
        for row in matrix:
            f.write(" ".join(f"{x:.6}" for x in row) + "\n")

def main():
    n = 2000
    print(f"Генерация...")
    generate_matrix(n, "matrix_a2000.txt")
    generate_matrix(n, "matrix_b2000.txt")
    print("Nice")

if __name__ == "__main__":
    main()