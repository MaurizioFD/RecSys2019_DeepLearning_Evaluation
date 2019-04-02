#ifndef UTILS_H
#define UTILS_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_sort.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_psi.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <memory.h>


#define outlog(format, args...) \
    fprintf(stderr, format, args); \
    fprintf(stderr, "\n");

int compare (const void * a, const void * b);

inline double safe_log(double x) {
  if (x <= 0)
    return(-10000);
  else 
    return(log(x));
}
double log_sum(double, double);

inline double vget(const gsl_vector* v, int i) { return(gsl_vector_get(v, i)); }

inline void vset(gsl_vector* v, int i, double x) { gsl_vector_set(v, i, x); }

// Increment a vector element by a double.
inline void vinc(gsl_vector* v, int i, double x) {
  vset(v, i, vget(v, i) + x);
}

inline double mget(const gsl_matrix* m, int i, int j)
{ return(gsl_matrix_get(m, i, j)); }

inline void mset(gsl_matrix* m, int i, int j, double x)
{ gsl_matrix_set(m, i, j, x); }

// Increment a matrix element by a double.
void minc(gsl_matrix*, int, int, double);

void col_sum(const gsl_matrix*, gsl_vector*);
void row_sum(const gsl_matrix*, gsl_vector*);

void vct_fprintf(FILE* file, const gsl_vector* v);
void mtx_fprintf(FILE* file, const gsl_matrix* m);
void mtx_fscanf(FILE* file, gsl_matrix* m);

inline bool check_sym(const gsl_matrix *m) {
  for (size_t i = 0; i < m->size1-1; i ++)
    for (size_t j=i; j < m->size2; j ++)
      if (mget(m, i, j) != mget(m, j, i)) {
        printf("not sym\n");
        return false;
      }
  return true;
}

double log_det(const gsl_matrix*);

void matrix_inverse(const gsl_matrix*, gsl_matrix*);
void matrix_vector_solve(const gsl_matrix* m, const gsl_vector* b, gsl_vector* v);

void sym_eigen(gsl_matrix*, gsl_vector*, gsl_matrix*);

inline double vsum(const gsl_vector* v) {
  double val = 0;
  int i, size = v->size;
  for (i = 0; i < size; i++)
    val += vget(v, i);
  return(val);
}

double vnorm(const gsl_vector * v);

void gsl_vector_apply(gsl_vector* x, double(*fun)(double));
void vct_log(gsl_vector* v);
void mtx_log(gsl_matrix* x);
void vct_exp(gsl_vector* x);
void mtx_exp(gsl_matrix* x);

double mahalanobis_distance(const gsl_matrix * m, const gsl_vector* u, const gsl_vector* v);
double mahalanobis_prod(const gsl_matrix * m, const gsl_vector* u, const gsl_vector* v);
double matrix_dot_prod(const gsl_matrix * m1, const gsl_matrix* m2);

void choose_k_from_n(int k, int n, int* result, int* src);

double log_normalize(gsl_vector* x);
double vnormalize(gsl_vector* x);

int dir_exists(const char *dname);
bool file_exists(const char * filename);
void make_directory(const char* name);

double  digamma(double x);
unsigned int rmultinomial(const gsl_vector* v);
double  rgamma(double a, double b);
double  rbeta(double a, double b);
unsigned int rbernoulli(double p);
double runiform();
void rshuffle (void* base, size_t n, size_t size);
unsigned long int runiform_int(unsigned long int n);

// new and free random number generator
gsl_rng* new_random_number_generator(long seed);
void free_random_number_generator(gsl_rng * random_number_generator);

#endif
