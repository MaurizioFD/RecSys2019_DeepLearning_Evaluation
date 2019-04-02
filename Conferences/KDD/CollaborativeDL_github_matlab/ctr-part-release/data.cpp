#include <assert.h>
#include <stdio.h>
#include "data.h"

c_data::c_data() {
}

c_data::~c_data() {
  for (size_t i = 0; i < m_vec_data.size(); i ++) {
    int* ids = m_vec_data[i];
    if (ids != NULL) delete [] ids;
  }
  m_vec_data.clear();
  m_vec_len.clear();
}

void c_data::read_data(const char * data_filename, int OFFSET) {

  int length = 0, n = 0, id = 0, total = 0;

  FILE * fileptr;
  fileptr = fopen(data_filename, "r");

  while ((fscanf(fileptr, "%10d", &length) != EOF)) {
    int * ids = NULL;
    if (length > 0) {
      ids = new int[length];
      for (n = 0; n < length; n++) {
        fscanf(fileptr, "%10d", &id);
        ids[n] = id - OFFSET;
      }
    }
    m_vec_data.push_back(ids);
    m_vec_len.push_back(length);
    total += length;
  }
  fclose(fileptr);
  printf("read %d vectors with %d entries ...\n", (int)m_vec_len.size(), total);
}

