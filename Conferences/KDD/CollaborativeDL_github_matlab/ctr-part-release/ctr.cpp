#include "ctr.h"
#include <cstring>

extern gsl_rng * RANDOM_NUMBER;
int min_iter = 1;

c_ctr::c_ctr() {
  m_theta = NULL;
  m_U = NULL;
  m_V = NULL;

  m_num_factors = 0; // m_num_topics
  m_num_items = 0; // m_num_docs
  m_num_users = 0; // num of users
}

c_ctr::~c_ctr() {
  // free memory
  if (m_theta != NULL) gsl_matrix_free(m_theta);
  if (m_U != NULL) gsl_matrix_free(m_U);
  if (m_V != NULL) gsl_matrix_free(m_V);
}

void c_ctr::read_init_information(const char* theta_init_path) {
  int num_topics = m_num_factors;
  m_theta = gsl_matrix_alloc(m_num_items, num_topics);
  printf("\nreading theta initialization from %s\n", theta_init_path);
  FILE * f = fopen(theta_init_path, "r");
  mtx_fscanf(f, m_theta);
  fclose(f);

  //normalize m_theta, in case it's not
  int is_nor = 0;
  if(is_nor==1){
    for (size_t j = 0; j < m_theta->size1; j ++) {
      gsl_vector_view theta_v = gsl_matrix_row(m_theta, j);
      vnormalize(&theta_v.vector);
    }
  }
}

void c_ctr::set_model_parameters(int num_factors, 
                                 int num_users, 
                                 int num_items) {
  m_num_factors = num_factors;
  m_num_users = num_users;
  m_num_items = num_items;
}

void c_ctr::init_model(int ctr_run) {

  m_U = gsl_matrix_calloc(m_num_users, m_num_factors);
  m_V = gsl_matrix_calloc(m_num_items, m_num_factors);
  // initialize eta+ //added

  if (ctr_run) {
    gsl_matrix_memcpy(m_V, m_theta);
  }
  else {
    // this is for convenience, so that updates are similar.
    m_theta = gsl_matrix_calloc(m_num_items, m_num_factors);
    size_t i;
    size_t j;
    for (i = 0; i < m_V->size1; i ++) 
      for (j = 0; j < m_V->size2; j ++){ 
        mset(m_V, i, j, runiform());
      }
  }
}

void c_ctr::learn_map_estimate(const c_data* users, const c_data* items,
                               const ctr_hyperparameter* param,
                               const char* directory) {
  // init model parameters
  printf("\ninitializing the model ...\n");
  init_model(param->ctr_run);
  // filename
  char name[500];

  // start time
  time_t start, current;
  time(&start);
  int elapsed = 0;

  int iter = 0;
  double likelihood_out;
  double likelihood = -exp(50), likelihood_old;
  double converge = 1.0;

  /// create the state log file 
  sprintf(name, "%s/state.log", directory);
  FILE* file = fopen(name, "w");
  fprintf(file, "iter time likelihood converge\n");

  /* alloc auxiliary variables */
  gsl_matrix* XX = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* A  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_matrix* B  = gsl_matrix_alloc(m_num_factors, m_num_factors);
  gsl_vector* x  = gsl_vector_alloc(m_num_factors);

  gsl_matrix* phi = NULL;
  gsl_matrix* word_ss = NULL;
  gsl_matrix* log_beta = NULL;
  gsl_vector* gamma = NULL;

  /* tmp variables for indexes */
  int i, j, m, n, l;
  int* item_ids; 
  int* user_ids;

  double result;

  /// confidence parameters
  double a_minus_b = param->a - param->b;

  // read init V from file: just for CDL
  // for standalone CTR, this part should be deleted
  printf("\nreading V initialization from %s/final-V.dat\n", directory);
  sprintf(name, "%s/final-V.dat", directory);
  FILE * file_init_V = fopen(name, "r");
  mtx_fscanf(file_init_V, m_V);
  fclose(file_init_V);

  while ((iter < param->max_iter and converge > 1e-6 ) or iter < min_iter) {


    likelihood_old = likelihood;
    likelihood = 0.0;
    likelihood_out = 0.0;

    // update U
    // XX: tmp variable
    gsl_matrix_set_zero(XX);
    // VV^T
    for (j = 0; j < m_num_items; j ++) {
      m = items->m_vec_len[j];
      if (m>0) {
        gsl_vector_const_view v = gsl_matrix_const_row(m_V, j); 
        gsl_blas_dger(1.0, &v.vector, &v.vector, XX);
      }
    }
    gsl_matrix_scale(XX, param->b);// b: negative item weight
    // this is only for U
    gsl_matrix_add_diagonal(XX, param->lambda_u); 

    for (i = 0; i < m_num_users; i ++) {
      item_ids = users->m_vec_data[i];
      n = users->m_vec_len[i];
      if (n > 0) {// for using sparsicity?
        // this user has rated some articles
        // a and b are the stage parameters in matrix C
        gsl_matrix_memcpy(A, XX);
        gsl_vector_set_zero(x);
        for (l=0; l < n; l ++) {
          j = item_ids[l];
          gsl_vector_const_view v = gsl_matrix_const_row(m_V, j); 
          gsl_blas_dger(a_minus_b, &v.vector, &v.vector, A); 
          gsl_blas_daxpy(param->a, &v.vector, x);
        }

        gsl_vector_view u = gsl_matrix_row(m_U, i);
        // A = V*C_i*V^T+\lambda_u*I_k
        // x = V*C_i*R_i
        matrix_vector_solve(A, x, &(u.vector));

        // update the likelihood
        gsl_blas_ddot(&u.vector, &u.vector, &result);
        likelihood += -0.5 * param->lambda_u * result;
        likelihood_out += -0.5 * param->lambda_u * result;
      }
    }

    // update V
    // clear word_ss
    if (param->ctr_run && param->theta_opt) gsl_matrix_set_zero(word_ss);
    // tmp XX
    // same as the U updating process
    gsl_matrix_set_zero(XX);
    for (i = 0; i < m_num_users; i ++) {
      n = users->m_vec_len[i]; 
      if (n>0) {
        gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);
        gsl_blas_dger(1.0, &u.vector, &u.vector, XX);
      }
    }
    gsl_matrix_scale(XX, param->b);//deleted

    for (j = 0; j < m_num_items; j ++) {
      gsl_vector_view v = gsl_matrix_row(m_V, j);
      gsl_vector_view theta_v = gsl_matrix_row(m_theta, j);

      user_ids = items->m_vec_data[j];
      m = items->m_vec_len[j];


      if (m>0) {
        // m > 0, some users have rated this article
        //
        gsl_matrix_memcpy(A, XX);
        gsl_vector_set_zero(x);
        for (l = 0; l < m; l ++) {
          i = user_ids[l];
          gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);  
          gsl_blas_dger(a_minus_b, &u.vector, &u.vector, A);
          gsl_blas_daxpy(param->a, &u.vector, x);
        }

        // adding the topic vector
        // even when ctr_run=0, m_theta=0
        gsl_blas_daxpy(param->lambda_v, &theta_v.vector, x);
        
        gsl_matrix_memcpy(B, A); // save for computing likelihood 

        // here different from U update
        gsl_matrix_add_diagonal(A, param->lambda_v);  
        matrix_vector_solve(A, x, &v.vector);//deleted


        // update the likelihood for the relevant part
        likelihood += -0.5 * m * param->a;
        likelihood_out += -0.5 * m * param->a;
        for (l = 0; l < m; l ++) {
          i = user_ids[l];
          gsl_vector_const_view u = gsl_matrix_const_row(m_U, i);  
          gsl_blas_ddot(&u.vector, &v.vector, &result);
          likelihood += param->a * result;
          likelihood_out += param->a * result;
        }
        // ?
        likelihood += -0.5 * mahalanobis_prod(B, &v.vector, &v.vector);
        likelihood_out += -0.5 * mahalanobis_prod(B, &v.vector, &v.vector);
        // likelihood part of theta, even when theta=0, which is a
        // special case
        gsl_vector_memcpy(x, &v.vector);
        gsl_vector_sub(x, &theta_v.vector);
        gsl_blas_ddot(x, x, &result);
        likelihood += -0.5 * param->lambda_v * result;
        likelihood_out += -0.5 * param->lambda_v * result;
        // finish updating V
      }
      else {
      // m=0, this article has never been rated
        gsl_matrix_memcpy(A, XX);
        gsl_vector_set_zero(x);

        // adding the topic vector
        // even when ctr_run=0, m_theta=0
        gsl_blas_daxpy(param->lambda_v, &theta_v.vector, x);
        
        gsl_matrix_memcpy(B, A); // save for computing likelihood 

        // here different from U update
        gsl_matrix_add_diagonal(A, param->lambda_v);  
        matrix_vector_solve(A, x, &v.vector);//deleted

        gsl_vector_memcpy(x, &v.vector);
        gsl_vector_sub(x, &theta_v.vector);
        gsl_blas_ddot(x, x, &result);
        likelihood += -0.5 * param->lambda_v * result;
        likelihood_out += -0.5 * param->lambda_v * result;
        // finish updating unrated part of V

        // update theta 
      }
    }

    time(&current);
    elapsed = (int)difftime(current, start);

    iter++;
    converge = fabs((likelihood-likelihood_old)/likelihood_old);

    if (likelihood < likelihood_old) printf("likelihood is decreasing!\n");

    fprintf(file, "%04d %06d %10.5f %.10f\n", iter, elapsed, likelihood, converge);
    fflush(file);

    // mine
    const char* respnt = strchr(directory,'p');

    printf("%s: iter=%04d, time=%06d, likelihood=%.5f/%.5f, converge=%.10f\n", respnt, iter, elapsed, likelihood, likelihood_out, converge);

    // save intermediate results
    if (iter % param->save_lag == 0) {

      sprintf(name, "%s/%04d-U.dat", directory, iter);
      FILE * file_U = fopen(name, "w");
      mtx_fprintf(file_U, m_U);
      fclose(file_U);

      sprintf(name, "%s/%04d-V.dat", directory, iter);
      FILE * file_V = fopen(name, "w");
      mtx_fprintf(file_V, m_V);
      fclose(file_V);
    }
  }

  // save final results
  sprintf(name, "%s/final-U.dat", directory);
  FILE * file_U = fopen(name, "w");
  mtx_fprintf(file_U, m_U);
  fclose(file_U);

  sprintf(name, "%s/final-V.dat", directory);
  FILE * file_V = fopen(name, "w");
  mtx_fprintf(file_V, m_V);
  fclose(file_V);

  sprintf(name, "%s/final-likelihood.dat", directory);
  FILE * file_L = fopen(name, "w");
  fprintf(file_L,"%f",likelihood_out);
  fclose(file_L);

  // free memory
  gsl_matrix_free(XX);
  gsl_matrix_free(A);
  gsl_matrix_free(B);
  gsl_vector_free(x);

  if (param->ctr_run && param->theta_opt) {
    gsl_matrix_free(phi);
    gsl_matrix_free(log_beta);
    gsl_matrix_free(word_ss);
    gsl_vector_free(gamma);
  }
}
