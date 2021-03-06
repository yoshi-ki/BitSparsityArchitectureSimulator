#include "PE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error
#include <omp.h>

namespace simulator
{
  PE::PE(){
    psum = 0;
  }

  int PE::execute_one_step(
    PEInput bitActivations,
    PEInput bitWeights
  )
  {
    // compute 16 multiplications
    // #pragma omp parallel for
    for (int i = 0; i < num_PE_parallel; i++){
      if (bitActivations.isValid[i] && bitWeights.isValid[i]){
        // TODO: change this to 22 bit integer
        auto multIsNegative = bitActivations.isNegative[i] ^ bitWeights.isNegative[i];
        auto multExp = bitActivations.bitInputValue[i] + bitWeights.bitInputValue[i];
        // #pragma omp atomic
        psum += multIsNegative ? (-(1 << multExp)) : (1 << multExp);
      }
    }
    return psum;
  }

  void PE::reset_state(){
    psum = 0;
  }

  int PE::get_psum()
  {
    return psum;
  }
}
