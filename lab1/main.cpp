#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;
bool read_matrix(const string& filename, vector<double>& matrix, int& n) {
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "Error: Can't read " << filename << '\n';
		return false;
	}
	file >> n;
	matrix.resize(n * n);
	for (int i = 0; i < n * n; ++i) {
		if (!(file >> matrix[i])) {
			cerr << "Error: not enough data in " << filename << '\n';
			return false;
		}
	}
	file.close();
	return true;
}

bool write_matrix(const string& filename, const vector<double>& matrix, int n) {
	ofstream file(filename);
	if (!file.is_open()) {
		cerr << "ERROR: Can't create " << filename << '\n';
		return false;
	}
	file << n << '\n';
	file << fixed << setprecision(6);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j){
			file << matrix[i * n + j] << " ";
		}
		file << '\n';
	}
	file.close();
	return true;
}

void multiply_matrix(const vector<double>& A, const vector<double>& B, vector<double>& C, int n) {
	fill(C.begin(), C.end(), 0.0);
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < n; ++i) {
		for (int k = 0; k < n; ++k) {
			double r = A[i * n + k];
			for (int j = 0; j < n; ++j) {
				C[i * n + j] += r * B[k * n + j];
			}
		}
	}
}

int main() {

	string fileA = "matrix_a.txt";
	string fileB = "matrix_b.txt";
	string fileC = "matrix_c.txt";
	
	vector<double> A, B, C;
	int nB = 50;
	int nA = nB;
	if (!read_matrix(fileA, A, nA) || !read_matrix(fileB, B, nB)) {
		return 1;
	}

	if (nA != nB) {
		cerr << "ERROR: Matricses should be square and of the same size" << '\n';
		return 1;
	}
	int n = nA;

	C.resize(n * n);

	auto start = chrono::high_resolution_clock::now();

	multiply_matrix(A, B, C, n);

	auto end = chrono::high_resolution_clock::now();
	chrono::duration<double> diff = end - start;

	if (!write_matrix(fileC, C, n)) {
		return 1;
	}

	double flops = 2.0 * n * n * n;
	cout << "Matrix size (N): " << n << '\n';
	cout << "FLOPs: " << scientific << flops << '\n';
	cout << "time: " << fixed << setprecision(6) << diff.count() << " sec." << '\n';
	cout << "GFLOPS: " << (flops / diff.count()) / 1e9 << '\n';
	cout << "OpenMP threads: " << omp_get_max_threads() << '\n';
	cout << "Saved in: " << fileC << '\n';
}