// class for ctr
//
#ifndef CTR_H
#define CTR_H

#include "utils.h"
#include "data.h"

struct ctr_hyperparameter {
  double a;
  double b;
  double lambda_u;
  double lambda_v;
  int    random_seed;
  int    max_iter;
  int    save_lag;
  int    theta_opt;
  int    ctr_run;
  
  void set(double aa, double bb, 
           double lu, double lv,
           int rs,    int mi, 
           int sl,    int to,
           int cr=1) {
    a = aa; b = bb; 
    lambda_u = lu; lambda_v = lv;
    random_seed = rs; max_iter = mi;
    save_lag = sl; theta_opt = to;
    ctr_run = cr;
  }

  void save(char* filename) {
    FILE * file = fopen(filename, "w");
    fprintf(file, "a = %.4f\n", a);
    fprintf(file, "b = %.4f\n", b);
    fprintf(file, "lambda_u = %.4f\n", lambda_u);
    fprintf(file, "lambda_v = %.4f\n", lambda_v);
    fprintf(file, "random seed = %d\n", (int)random_seed);
    fprintf(file, "max iter = %d\n", max_iter);
    fprintf(file, "save lag = %d\n", save_lag);
    fprintf(file, "theta opt = %d\n", theta_opt);
    fprintf(file, "ctr run = %d\n", ctr_run);
    fclose(file);
  }
};

class c_ctr {
public:
  c_ctr();
  ~c_ctr();
  void read_init_information(const char* theta_init_path);

  void set_model_parameters(int num_factors, 
                            int num_users, 
                            int num_items);

  void learn_map_estimate(const c_data* users, const c_data* items, 
                          const ctr_hyperparameter* param,
                          const char* directory);
  void init_model(int ctr_run);

public:
  gsl_matrix* m_theta;

  gsl_matrix* m_U;
  gsl_matrix* m_V;

  int m_num_factors; // m_num_topics
  int m_num_items; // m_num_docs
  int m_num_users; // num of users
};

#endif // CTR_H
