#include "utils.h"

extern gsl_rng * RANDOM_NUMBER;

/*
 * compare two ints
 * */

int compare (const void * a, const void * b) {
  return ( *(int*)a - *(int*)b );
}

/*
 * given log(a) and log(b), return log(a+b)
 *
 */

double log_sum(double log_a, double log_b) {
  double v;

  if (log_a == -1) return(log_b);

  if (log_a < log_b) {
    v = log_b+log(1 + exp(log_a-log_b));
  }
  else {
    v = log_a+log(1 + exp(log_b-log_a));
  }
  return(v);
}

/*
void vinc(gsl_vector* v, int i, double x)
{
    vset(v, i, vget(v, i) + x);
}
*/

void minc(gsl_matrix* m, int i, int j, double x) {
    mset(m, i, j, mget(m, i, j) + x);
}

/*
 * compute the row sums of a matrix
 *
 */

void row_sum(const gsl_matrix* m, gsl_vector* val) {
  size_t i, j;
  gsl_vector_set_zero(val);

  for (i = 0; i < m->size1; i++)
    for (j = 0; j < m->size2; j++)
      vinc(val, i, mget(m, i, j));
}

/*
 * compute the column sums of a matrix
 *
 */

void col_sum(const gsl_matrix* m, gsl_vector* val) {
  size_t i, j;
  gsl_vector_set_zero(val);

  for (i = 0; i < m->size1; i++)
    for (j = 0; j < m->size2; j++)
      vinc(val, j, mget(m, i, j));
}


/*
 * print a vector to standard out
 *
 */

void vct_fprintf(FILE * file, const gsl_vector * v) {
  size_t i;
  for (i = 0; i < v->size; i++)
    fprintf(file, "%10.8e ", vget(v, i));
    fprintf(file, "\n");
}


/*
 * print a matrix to standard out
 *
 */

void mtx_fprintf(FILE * file, const gsl_matrix * m) {
  size_t i, j;
  for (i = 0; i < m->size1; i++) {
    for (j = 0; j < m->size2; j++)
      fprintf(file, "%10.8e ", mget(m, i, j));// changed
    fprintf(file, "\n");
  }
}

void mtx_fscanf(FILE* file, gsl_matrix* m) {
  size_t i, j;
  double x;
  for (i = 0; i < m->size1; i++) {
    for (j = 0; j < m->size2; j++) {
      fscanf(file, "%lf", &x);
      mset(m, i, j, x);
    }
  }
}

/*
 * matrix vector solve using blas
 *
 */

void matrix_vector_solve(const gsl_matrix* m, const gsl_vector* b, gsl_vector* v) {
  gsl_matrix *lu;
  gsl_permutation* p;
  int signum;

  p = gsl_permutation_alloc(m->size1);
  lu = gsl_matrix_alloc(m->size1, m->size2);

  gsl_matrix_memcpy(lu, m);
  gsl_linalg_LU_decomp(lu, p, &signum);
  gsl_linalg_LU_solve(lu, p, b, v);

  gsl_matrix_free(lu);
  gsl_permutation_free(p);
}

/*
 * matrix inversion using blas
 *
 */

void matrix_inverse(const gsl_matrix* m, gsl_matrix* inverse) {
  gsl_matrix *lu;
  gsl_permutation* p;
  int signum;

  p = gsl_permutation_alloc(m->size1);
  lu = gsl_matrix_alloc(m->size1, m->size2);

  gsl_matrix_memcpy(lu, m);
  gsl_linalg_LU_decomp(lu, p, &signum);
  gsl_linalg_LU_invert(lu, p, inverse);

  gsl_matrix_free(lu);
  gsl_permutation_free(p);
}

/*
 * log determinant using blas
 *
 */

double log_det(const gsl_matrix* m) {
  gsl_matrix* lu;
  gsl_permutation* p;
  double result;
  int signum;

  p = gsl_permutation_alloc(m->size1);
  lu = gsl_matrix_alloc(m->size1, m->size2);

  gsl_matrix_memcpy(lu, m);
  gsl_linalg_LU_decomp(lu, p, &signum);
  result = gsl_linalg_LU_lndet(lu);

  gsl_matrix_free(lu);
  gsl_permutation_free(p);

  return(result);
}


/*
 * eigenvalues of a symmetric matrix using blas
 *
 */

void sym_eigen(gsl_matrix* m, gsl_vector* vals, gsl_matrix* vects) {
  gsl_eigen_symmv_workspace* wk;
  gsl_matrix* mcpy;
  int r;

  mcpy = gsl_matrix_alloc(m->size1, m->size2);
  wk = gsl_eigen_symmv_alloc(m->size1);
  gsl_matrix_memcpy(mcpy, m);
  r = gsl_eigen_symmv(mcpy, vals, vects, wk);
  gsl_eigen_symmv_free(wk);
  gsl_matrix_free(mcpy);
}


/*
 * sum of a vector
 *
 */
/*
double sum(const gsl_vector* v) {
  double val = 0;
  int i, size = v->size;
  for (i = 0; i < size; i++)
    val += vget(v, i);
  return(val);
}
*/

/*
 * apply a function to each element of a gsl vector.
 *
 */
void gsl_vector_apply(gsl_vector* x, double(*fun)(double)) {
  size_t i;
  for(i = 0; i < x->size; i ++)
    vset(x, i, fun(vget(x, i)));
}


/*
 * take log of each element for a vector
 *
 */
void vct_log(gsl_vector* v) {
  int i, size = v->size;
  for (i = 0; i < size; i++)
    vset(v, i, safe_log(vget(v, i)));
}


/*
 * take log of each element for a matrix
 *
 */
void mtx_log(gsl_matrix* x) {
  size_t i, j;
  for (i = 0; i < x->size1; i++)
    for (j = 0; j < x->size2; j++)
      mset(x, i, j, safe_log(mget(x, i, j)));
}



/*
 * l2 norm of a vector
 *
 */

double vnorm(const gsl_vector *v) {
  return gsl_blas_dnrm2(v);
}


/*
 * normalize a vector in log space
 *
 * x_i = log(a_i)
 * v = log(a_1 + ... + a_k)
 * x_i = x_i - v
 *
 */

double log_normalize(gsl_vector* x) {
  double v = vget(x, 0);
  size_t i;

  for (i = 1; i < x->size; i++)
    v = log_sum(v, vget(x, i));

  for (i = 0; i < x->size; i++)
    vset(x, i, vget(x,i)-v);

  return v;
}


/*
 * normalize a positive vector
 *
 */

double vnormalize(gsl_vector* x) {
  double v = vsum(x);
  if (v > 0 || v < 0)
    gsl_vector_scale(x, 1/v);
  return v;
}


/*
 * exponentiate a vector
 *
 */

void vct_exp(gsl_vector* x) {
  for (size_t i = 0; i < x->size; i++)
    vset(x, i, exp(vget(x, i)));
}

/*
 * exponentiate a matrix
 *
 */
void mtx_exp(gsl_matrix* x) {
  size_t i, j;
  for (i = 0; i < x->size1; i++)
    for (j = 0; j < x->size2; j++)
      mset(x, i, j, exp(mget(x, i, j)));
}

double mahalanobis_distance(const gsl_matrix * m, 
                            const gsl_vector* u, 
                            const gsl_vector* v) {
  double val = 0;
  gsl_vector* x = gsl_vector_alloc(u->size);
  gsl_vector_memcpy(x, u);
  gsl_vector_sub(x, v);
  val = mahalanobis_prod(m, x, x);
  gsl_vector_free(x);
  return val;
}

// blasified
double mahalanobis_prod(const gsl_matrix * m, 
                        const gsl_vector* u, 
                        const gsl_vector* v) {
  gsl_vector* x = gsl_vector_alloc(u->size);
  gsl_blas_dgemv(CblasNoTrans, 1.0, m, v, 0.0, x);
  double val = 0;
  gsl_blas_ddot(u, x, &val);
  gsl_vector_free(x);
  return val;
}

double matrix_dot_prod(const gsl_matrix * m1, const gsl_matrix* m2) {
  double val = 0, result;
  for (size_t i = 0; i < m1->size1; i ++) {
    gsl_vector_const_view v1 = gsl_matrix_const_row(m1, i);
    gsl_vector_const_view v2 = gsl_matrix_const_row(m2, i);
    gsl_blas_ddot(&v1.vector, &v2.vector, &result); 
    val += result;
  }
  return val;
}


/**
*
* check if file exisits
*/
bool file_exists(const char * filename) {
  if ( 0 == access(filename, R_OK))
    return true;
  return false;
}



/*
 * check if a directory exists
 *
 * !!! shouldn't be here
 */

int dir_exists(const char *dname) {
  struct stat st;
  int ret;

  if (stat(dname,&st) != 0) {
    return 0;
  }

  ret = S_ISDIR(st.st_mode);

  if(!ret) {
    errno = ENOTDIR;
  }

  return ret;
}

void make_directory(const char* name) {
  mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
}


/*
 * new random number generator
 *
 */
gsl_rng * new_random_number_generator(long seed) {
  gsl_rng * random_number_generator = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(random_number_generator, (long) seed); // init the seed

  return random_number_generator;
}

/* 
 * free random number generator
 * */

void free_random_number_generator(gsl_rng * random_number_generator) {
  gsl_rng_free(random_number_generator);
}

void choose_k_from_n(int k, int n, int* result, int* src) {
  gsl_ran_choose (RANDOM_NUMBER, (void *) result,  k, (void *) src, n, sizeof(int));
}

double digamma(double x) {
  return gsl_sf_psi(x);
}

unsigned int rmultinomial(const gsl_vector* v) {
  size_t i;

  double sum = vsum(v);

  double u = runiform() * sum;
  double cum_sum = 0.0;
  for (i = 0; i < v->size; i ++)
  {
      cum_sum += vget(v, i);
      if (u < cum_sum) break;
  }
  return i;
}

double rgamma(double a, double b) {
  return gsl_ran_gamma_mt(RANDOM_NUMBER, a, b);
}

double rbeta(double a, double b) {
  return gsl_ran_beta(RANDOM_NUMBER, a, b);
}

unsigned int rbernoulli(double p) {
  return gsl_ran_bernoulli(RANDOM_NUMBER, p);
}

double runiform() {
  return gsl_rng_uniform_pos(RANDOM_NUMBER);
}

void rshuffle(void* base, size_t n, size_t size) {
  gsl_ran_shuffle(RANDOM_NUMBER, base, n, size);
}

unsigned long int runiform_int(unsigned long int n) {
  return  gsl_rng_uniform_int(RANDOM_NUMBER, n);
}

