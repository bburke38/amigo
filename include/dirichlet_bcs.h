#ifndef AMIGO_DIRICHLET_BCS_H
#define AMIGO_DIRICHLET_BCS_H

#include <memory>

#include "amigo.h"
#include "csr_matrix.h"
#include "vector.h"

namespace amigo {

class DirichletBCs {
 public:
  template <typename T>
  DirichletBCs(int num_bcs, int bcs[], std::shared_ptr<CSRMat<T>> mat) {}

  void zero_rows(std::shared_ptr<Vector<T>> vec) {}

  void zero_rows_and_columns(std::shared_ptr<CSRMat<T>> mat) {}
};

}  // namespace amigo