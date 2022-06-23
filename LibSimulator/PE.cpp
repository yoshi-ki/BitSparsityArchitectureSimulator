#include "PE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

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
    for (int i = 0; i < num_PE_parallel; i++){
      // comment out for optimization
      // if (bitActivations.bitInputValue[i] >= 8 || bitWeights.bitInputValue[i] >= 8){
      //   throw std::runtime_error("error!");
      // }

      if (bitActivations.isValid[i] && bitWeights.isValid[i]){
        // TODO: change this to 22 bit integer
        auto multIsNegative = bitActivations.isNegative[i] ^ bitWeights.isNegative[i];
        auto multExp = bitActivations.bitInputValue[i] + bitWeights.bitInputValue[i];
        psum += multIsNegative ? (-(1 << multExp)) : (1 << multExp);
        // std::cout << psum << std::endl;
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
