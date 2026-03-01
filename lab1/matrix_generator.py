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
    n = 50
    print(f"Генерация...")
    generate_matrix(n, "matrix_a.txt")
    generate_matrix(n, "matrix_b.txt")
    print("Nice")

if __name__ == "__main__":
    main()