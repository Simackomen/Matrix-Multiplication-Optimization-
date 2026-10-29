/**********************************************************************************************************************
 * 
 * - Program task: explore different matrix multiplication approaches and compare their respective performance for 
 *                 random floating-point value square matrices of different dimensions. 
 * 
 * - Data structure: Matrices are stored as flat 1D arrays for better cache locality, when compared to the naive 
 *                   array-of-pointers implementation. 
 * 
 * - Compile with: clang -O3 mat_mul.c -o mat_mul.o
 * - Run with: ./mat_mul.o
 * 
 ***********************************************************************************************************************/

// Libraries
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<time.h>
#include <stdint.h>

// Helper function to generate a random floating-point value within a given range
// ARGUMENTS: 
// - double lo --> lower bound for random number generation
// - double hi --> upper bound for random number generation
// RETURN TYPE: double --> random floating-point value within the given range
static inline double random_float(double lo, double hi) {
    return lo + (hi - lo) * ((double)arc4random() / ((double)UINT32_MAX + 1.0));
}

// Helper function to randomize matrix values
// ARGUMENTS: 
// - double* A --> pointer to square matrix stored as a flat 1D array
// - size_t n --> dimension of the square matrix 
// - double min --> lower bound for random number generation
// - double max --> upper bound for random number generation
// RETURN TYPE: none
static inline void randomize(double* A, size_t n, float min, float max)
{
    for(size_t i = 0; i < n * n; i++) A[i] = random_float(min, max);
}

// Helper function to access a particular index of a matrix stored as a flat 1D array
// ARGUMENTS: 
// - size_t n --> square matrix dimension
// - size_t i --> row index of the entry
// - size_t j --> column index of the entry
// RETURN TYPE: size_t --> array index corresponding to the specified matrix position
static inline size_t idx(size_t n, size_t i, size_t j) {return n * i + j;}

// Helper function to print the matrix contents 
// ARGUMENTS: 
// - double* A --> square matrix stored as a flat 1D array
// - size_t n --> dimension of the square matrix
// RETURN TYPE: none
static inline void print_mat(double* A, size_t n)
{
    for(size_t i = 0; i < n; i++)
    {
        for(size_t j = 0; j < n; j++)
        {
            // Print values with 6 decimal places of precision
            printf("%10.6lf ", A[idx(n, i, j)]);
        }
        printf("\n");
    }
    printf("\n");
}

// Helper function to initialize a flat 1D array with zeroes
// ARGUMENTS: 
// - size_t n --> dimension of the square matrix
// RETURN TYPE: double pointer to a zero matrix, represented as a flat 1D array
static inline double* make_mat(size_t n) {return calloc(n*n, sizeof(double));}

// Helper function to deallocate a matrix
// ARGUMENTS:
// - double* A --> matrix pointer 
// RETURN TYPE: none
static inline void kill_mat(double* A) {free(A);}

// Helper function to read a matrix in from user
// ARGUMENTS: 
// - double* A --> pointer to an initialized matrix
// - size_t n --> dimension of the square matrix 
// RETURN TYPE: none
static inline void read_mat(double* A, size_t n)
{
    for(size_t i = 0; i < n * n; i++) scanf("%lf", &A[i]);
}

// Multiply two matrices together using the naive approach of the standard definition
// ARGUMENTS: 
// - double* A --> pointer to an initialized matrix A
// - double* B --> pointer to an initialized matrix B
// - size_t n --> dimension of both square matrices
// RETURN TYPE: double* --> product of matrix multiplication C
static inline double* matmul_naive(double* A, double* B, size_t n)
{
    // Create a blank matrix
    double* C = make_mat(n); 

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            for (size_t k = 0; k < n; k++) {
                C[idx(n, i, j)] += A[idx(n, i, k)] * B[idx(n, k, j)];
            }
        }
    }

    return C;
}

// Function to transpose a given matrix into another blank matrix
// (Original matrix will not be affected)
// ARGUMENTS: 
// - double* A --> matrix to transpose
// - size_t n --> dimension of the square matrix
// RETURN TYPE: double* --> transposed matrix T
double* transpose(double* A, size_t n) {
    double *T = make_mat(n);

    for (size_t i = 0; i < n; i++) 
    {
        for (size_t j = 0; j < n; j++) 
        {
            T[idx(n, j, i)] = A[idx(n, i, j)];
        }
    }

    return T;
}

// Multiply two matrices together using the improved approach of the standard definition
// Matrix B has been transposed for better cache-locality 
// ARGUMENTS: 
// - double* A --> pointer to an initialized matrix A
// - double* T --> pointer to an initialized matrix B
// - size_t n --> dimension of both square matrices
// RETURN TYPE: double* --> product of matrix multiplication C
static inline double* matmul_trans(double* A, double* T, size_t n)
{
    // Create a blank matrix
    double* C = make_mat(n); 

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            const double *a_row  = &A[i * n];
            const double *t_row = &T[j * n];
            for (size_t k = 0; k < n; k++)
            { 
                C[idx(n, i, j)] += a_row[k] * t_row[k];
            }
        }
    }

    return C;
}

/******************************************* BLOCKING APPROACH ********************************************/

#define BLOCK_THRESHOLD (1 << 3)  // switch to naive below this size

/* mat_add, mat_sub, and mat_mul_naive are unchanged from before */

static void mat_add(const double *A, size_t strideA,
                     const double *B, size_t strideB,
                     double *C, size_t strideC, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            C[i*strideC + j] = A[i*strideA + j] + B[i*strideB + j];
}

static void mat_mul_naive(const double *A, size_t strideA,
                           const double *B, size_t strideB,
                           double *C, size_t strideC, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++)
                sum += A[i*strideA + k] * B[k*strideB + j];
            C[i*strideC + j] = sum;
        }
}

/*
 * Recursive blocking engine: partitions A, B, C into quadrants and
 * computes each output quadrant as the sum of two sub-products
 * (the standard 8-multiply/4-add definition), recursing until the
 * submatrix size drops to BLOCK_THRESHOLD, then uses the naive kernel.
 */
static void block_rec(const double *A, size_t strideA,
                       const double *B, size_t strideB,
                       double *C, size_t strideC, size_t n) {
    if (n <= BLOCK_THRESHOLD) {
        mat_mul_naive(A, strideA, B, strideB, C, strideC, n);
        return;
    }

    size_t m = n / 2;

    const double *A11 = A,              *A12 = A + m;
    const double *A21 = A + m*strideA,  *A22 = A + m*strideA + m;
    const double *B11 = B,              *B12 = B + m;
    const double *B21 = B + m*strideB,  *B22 = B + m*strideB + m;
    double       *C11 = C,              *C12 = C + m;
    double       *C21 = C + m*strideC,  *C22 = C + m*strideC + m;

    double *T = malloc(m * m * sizeof(double));  // scratch for the 2nd product per quadrant

    /* C11 = A11*B11 + A12*B21 */
    block_rec(A11, strideA, B11, strideB, C11, strideC, m);
    block_rec(A12, strideA, B21, strideB, T, m, m);
    mat_add(C11, strideC, T, m, C11, strideC, m);

    /* C12 = A11*B12 + A12*B22 */
    block_rec(A11, strideA, B12, strideB, C12, strideC, m);
    block_rec(A12, strideA, B22, strideB, T, m, m);
    mat_add(C12, strideC, T, m, C12, strideC, m);

    /* C21 = A21*B11 + A22*B21 */
    block_rec(A21, strideA, B11, strideB, C21, strideC, m);
    block_rec(A22, strideA, B21, strideB, T, m, m);
    mat_add(C21, strideC, T, m, C21, strideC, m);

    /* C22 = A21*B12 + A22*B22 */
    block_rec(A21, strideA, B12, strideB, C22, strideC, m);
    block_rec(A22, strideA, B22, strideB, T, m, m);
    mat_add(C22, strideC, T, m, C22, strideC, m);

    free(T);
}

/*
 * Public entry point: allocates a fresh, tightly-packed n x n matrix C
 * (stride n), fills it with A * B via recursive blocking, and returns it.
 * Caller is responsible for freeing the returned pointer.
 */
double *block_multiply(const double *A, size_t strideA,
                        const double *B, size_t strideB,
                        size_t n) {
    double *C = malloc(n * n * sizeof(double));
    if (!C) return NULL;

    block_rec(A, strideA, B, strideB, C, n, n);

    return C;
}

/* Thin wrapper: A and B are standalone n x n matrices (stride == n). */
double* matmul_block(double *A, double *B, size_t n) {
    return block_multiply(A, n, B, n, n);
}

/******************************************* STRASSEN'S APPROACH ********************************************/

#define STRASSEN_THRESHOLD (1 << 3) // switch to naive below this size

static inline void mat_sub(const double *A, size_t strideA,
                     const double *B, size_t strideB,
                     double *C, size_t strideC, size_t n) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            C[i*strideC + j] = A[i*strideA + j] - B[i*strideB + j];
}

/* Internal recursive engine: still writes into a caller-provided C,
   since quadrant views of C need their own stride during recursion. */
static inline void strassen_rec(const double *A, size_t strideA,
                          const double *B, size_t strideB,
                          double *C, size_t strideC, size_t n) {
    if (n <= STRASSEN_THRESHOLD) {
        mat_mul_naive(A, strideA, B, strideB, C, strideC, n);
        return;
    }

    size_t m = n / 2;

    const double *A11 = A,              *A12 = A + m;
    const double *A21 = A + m*strideA,  *A22 = A + m*strideA + m;
    const double *B11 = B,              *B12 = B + m;
    const double *B21 = B + m*strideB,  *B22 = B + m*strideB + m;
    double       *C11 = C,              *C12 = C + m;
    double       *C21 = C + m*strideC,  *C22 = C + m*strideC + m;

    double *M1 = malloc(m*m*sizeof(double));
    double *M2 = malloc(m*m*sizeof(double));
    double *M3 = malloc(m*m*sizeof(double));
    double *M4 = malloc(m*m*sizeof(double));
    double *M5 = malloc(m*m*sizeof(double));
    double *M6 = malloc(m*m*sizeof(double));
    double *M7 = malloc(m*m*sizeof(double));
    double *T1 = malloc(m*m*sizeof(double));
    double *T2 = malloc(m*m*sizeof(double));

    mat_add(A11, strideA, A22, strideA, T1, m, m);
    mat_add(B11, strideB, B22, strideB, T2, m, m);
    strassen_rec(T1, m, T2, m, M1, m, m);                 // M1 = (A11+A22)(B11+B22)

    mat_add(A21, strideA, A22, strideA, T1, m, m);
    strassen_rec(T1, m, B11, strideB, M2, m, m);           // M2 = (A21+A22)B11

    mat_sub(B12, strideB, B22, strideB, T2, m, m);
    strassen_rec(A11, strideA, T2, m, M3, m, m);           // M3 = A11(B12-B22)

    mat_sub(B21, strideB, B11, strideB, T2, m, m);
    strassen_rec(A22, strideA, T2, m, M4, m, m);           // M4 = A22(B21-B11)

    mat_add(A11, strideA, A12, strideA, T1, m, m);
    strassen_rec(T1, m, B22, strideB, M5, m, m);           // M5 = (A11+A12)B22

    mat_sub(A21, strideA, A11, strideA, T1, m, m);
    mat_add(B11, strideB, B12, strideB, T2, m, m);
    strassen_rec(T1, m, T2, m, M6, m, m);                  // M6 = (A21-A11)(B11+B12)

    mat_sub(A12, strideA, A22, strideA, T1, m, m);
    mat_add(B21, strideB, B22, strideB, T2, m, m);
    strassen_rec(T1, m, T2, m, M7, m, m);                  // M7 = (A12-A22)(B21+B22)

    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < m; j++) {
            C11[i*strideC+j] = M1[i*m+j] + M4[i*m+j] - M5[i*m+j] + M7[i*m+j];
            C12[i*strideC+j] = M3[i*m+j] + M5[i*m+j];
            C21[i*strideC+j] = M2[i*m+j] + M4[i*m+j];
            C22[i*strideC+j] = M1[i*m+j] - M2[i*m+j] + M3[i*m+j] + M6[i*m+j];
        }

    free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7);
    free(T1); free(T2);
}

/*
 * Public entry point: allocates a fresh, tightly-packed n x n matrix C
 * (stride n), fills it with A * B via Strassen's algorithm, and returns it.
 * Caller is responsible for freeing the returned pointer.
 */
static inline double *strassen(const double *A, size_t strideA,
                  const double *B, size_t strideB,
                  size_t n) {
    double *C = malloc(n * n * sizeof(double));
    if (!C) return NULL;

    strassen_rec(A, strideA, B, strideB, C, n, n);

    return C;
}

/*
 * Thin wrapper: A and B are standalone n x n matrices (stride == n).
 */
static inline double* matmul_strass(double *A, double *B, size_t n) {
    return strassen(A, n, B, n, n);
}

/******************************************* CACHE-OBLIVIOUS PARTITION ********************************************/

// Recursively partitions A (n x n, standard row/col layout) into T,
// writing 1x1 elements in recursive block order.
// idx tracks the next write position in T across recursive calls.
static inline void partition(const double *A, double *T, size_t n,
                      size_t x1, size_t y1, size_t x2, size_t y2,
                      size_t *idx)
{
    size_t xnu = (x1 + x2) / 2;
    size_t ynu = (y1 + y2) / 2;

    if (xnu > x1) {
        partition(A, T, n, x1,  y1,  xnu, ynu, idx);  // TL
        partition(A, T, n, xnu, y1,  x2,  ynu, idx);  // TR
        partition(A, T, n, x1,  ynu, xnu, y2,  idx);  // BL
        partition(A, T, n, xnu, ynu, x2,  y2,  idx);  // BR
    } else {
        T[*idx] = A[n * y1 + x1];
        (*idx)++;
    }
}

// Convenience wrapper: transforms A (n x n) into recursive-block layout T.
// T must already be allocated with n*n doubles.
void to_recursive_layout(const double *A, double *T, size_t n)
{
    size_t idx = 0;
    partition(A, T, n, 0, 0, n, n, &idx);
}

static void multiply_4x4_block(const double *A, const double *B, double *C)
{
    C[0]  += A[0]*B[0]  + A[2]*B[1]  + A[8]*B[4]   + A[10]*B[5];
    C[2]  += A[0]*B[2]  + A[2]*B[3]  + A[8]*B[6]   + A[10]*B[7];
    C[8]  += A[0]*B[8]  + A[2]*B[9]  + A[8]*B[12]  + A[10]*B[13];
    C[10] += A[0]*B[10] + A[2]*B[11] + A[8]*B[14]  + A[10]*B[15];

    C[1]  += A[1]*B[0]  + A[3]*B[1]  + A[9]*B[4]   + A[11]*B[5];
    C[3]  += A[1]*B[2]  + A[3]*B[3]  + A[9]*B[6]   + A[11]*B[7];
    C[9]  += A[1]*B[8]  + A[3]*B[9]  + A[9]*B[12]  + A[11]*B[13];
    C[11] += A[1]*B[10] + A[3]*B[11] + A[9]*B[14]  + A[11]*B[15];

    C[4]  += A[4]*B[0]  + A[6]*B[1]  + A[12]*B[4]  + A[14]*B[5];
    C[6]  += A[4]*B[2]  + A[6]*B[3]  + A[12]*B[6]  + A[14]*B[7];
    C[12] += A[4]*B[8]  + A[6]*B[9]  + A[12]*B[12] + A[14]*B[13];
    C[14] += A[4]*B[10] + A[6]*B[11] + A[12]*B[14] + A[14]*B[15];

    C[5]  += A[5]*B[0]  + A[7]*B[1]  + A[13]*B[4]  + A[15]*B[5];
    C[7]  += A[5]*B[2]  + A[7]*B[3]  + A[13]*B[6]  + A[15]*B[7];
    C[13] += A[5]*B[8]  + A[7]*B[9]  + A[13]*B[12] + A[15]*B[13];
    C[15] += A[5]*B[10] + A[7]*B[11] + A[13]*B[14] + A[15]*B[15];
}

static void multiply_8x8_block(const double *A, const double *B, double *C)
{
    // h = 4, so each quadrant is a 4x4 = 16-element contiguous block
    const double *A11 = A,    *A12 = A + 16, *A21 = A + 32, *A22 = A + 48;
    const double *B11 = B,    *B12 = B + 16, *B21 = B + 32, *B22 = B + 48;
    double       *C11 = C,    *C12 = C + 16, *C21 = C + 32, *C22 = C + 48;

    multiply_4x4_block(A11, B11, C11);
    multiply_4x4_block(A12, B21, C11);   // C11 += A12*B21

    multiply_4x4_block(A11, B12, C12);
    multiply_4x4_block(A12, B22, C12);

    multiply_4x4_block(A21, B11, C21);
    multiply_4x4_block(A22, B21, C21);

    multiply_4x4_block(A21, B12, C22);
    multiply_4x4_block(A22, B22, C22);
}

void recursive_multiply(const double *A, const double *B, double *C, size_t n)
{
    if (n == 8) {
        multiply_8x8_block(A, B, C);
        return;
    }
    if (n == 4) {
        multiply_4x4_block(A, B, C);
        return;
    }

    size_t h = n / 2;
    size_t block = h * h;

    const double *A11=A, *A12=A+block, *A21=A+2*block, *A22=A+3*block;
    const double *B11=B, *B12=B+block, *B21=B+2*block, *B22=B+3*block;
    double       *C11=C, *C12=C+block, *C21=C+2*block, *C22=C+3*block;

    recursive_multiply(A11,B11,C11,h);
    recursive_multiply(A12,B21,C11,h);

    recursive_multiply(A11,B12,C12,h);
    recursive_multiply(A12,B22,C12,h);

    recursive_multiply(A21,B11,C21,h);
    recursive_multiply(A22,B21,C21,h);

    recursive_multiply(A21,B12,C22,h);
    recursive_multiply(A22,B22,C22,h);
}

static inline double* matmul_partition(double *A, double *B, size_t n)
{
    double *C = calloc(n * n, sizeof(double));
    if (!C) {
        return NULL;
    }

    recursive_multiply(A, B, C, n);

    return C;
}

/******************************************* TIMING ALGORITHM ********************************************/

// Function pointer type matching the signature of the algorithm to be timed
typedef double* (*func)(double* A, double* B, size_t n);

// Times `num_runs` calls to `algo`, returns the average elapsed time in seconds.
// The result pointer from the final run is written to *out_result (can be NULL to ignore).
static inline double time_algorithm(func algo, double* A, double* B, size_t n,
                       size_t num_runs, size_t warm_up) {
    struct timespec start, end;
    double total_elapsed = 0.0;

    for(size_t w = 0; w < warm_up; w++) kill_mat(algo(A, B, n));

    for (size_t run = 0; run < num_runs; run++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        kill_mat(algo(A, B, n));
        clock_gettime(CLOCK_MONOTONIC, &end);

        total_elapsed += (end.tv_sec - start.tv_sec)  + (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    return total_elapsed / num_runs;
}

void run_trials(func algo)
{
    size_t dim = 1 << 4; 
    double* A = make_mat(dim); 
    double* B = make_mat(dim);
    randomize(A, dim, -100, 100);
    randomize(B, dim, -100, 100);

    printf("Avg runtime at input size %lu: %.9lf sec.\n", dim, time_algorithm(algo, A, B, dim, 500, 250));

    kill_mat(A);
    kill_mat(B);

    dim = 1 << 7;
    A = make_mat(dim); 
    B = make_mat(dim);
    randomize(A, dim, -100, 100);
    randomize(B, dim, -100, 100);

    printf("Avg runtime at input size %lu: %.9lf sec.\n", dim, time_algorithm(algo, A, B, dim, 100, 50));

    kill_mat(A);
    kill_mat(B);

    dim = 1 << 10; 
    A = make_mat(dim); 
    B = make_mat(dim);
    randomize(A, dim, -100, 100);
    randomize(B, dim, -100, 100);

    printf("Avg runtime at input size %lu: %.9lf sec.\n", dim, time_algorithm(algo, A, B, dim, 20, 10));
    kill_mat(A);
    kill_mat(B);

}

int main()
{
    printf("\nNaive approach of the standard matrix multiplication algorithm: \n");
    run_trials(matmul_naive);

    printf("\nImproved approach of the standard algorithm with transposition: \n");
    run_trials(matmul_trans);

    printf("\nRecursive blocking approach of the standard definition: \n");
    run_trials(matmul_block);

    printf("\nStrassen's recursive approach: \n");
    run_trials(matmul_strass);

    printf("\nCache-oblivious recursive partitioning approach: \n");
    run_trials(matmul_partition);
    return 0;
}