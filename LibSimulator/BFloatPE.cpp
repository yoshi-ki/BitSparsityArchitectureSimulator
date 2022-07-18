#include "BFloatPE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error
#include <omp.h>

namespace simulator
{
  BFloatPE::BFloatPE(){
    psum = 0;
  }

  int BFloatPE::execute_one_step(
    PEInput bitActivations,
    PEInput bitWeights,
    int psumShiftedWidth
  )
  {
    psum = psum >> psumShiftedWidth;
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

  void BFloatPE::reset_state(){
    psum = 0;
  }

  std::pair<int,int> BFloatPE::get_psum(int inputExp, int weightExp)
  {
    // output bfloat values
    std::cout << inputExp << " " << weightExp << " " << psum << std::endl;
    // TODO: shift mantissa part
    return std::pair<int,int>{inputExp+weightExp-127, psum};
  }
}
