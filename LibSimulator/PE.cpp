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
    for (int i = 0; i < bitActivations.bitInputValue.size(); i++){
      if (bitActivations.bitInputValue[i] >= 8 || bitWeights.bitInputValue[i] >= 8){
        throw std::runtime_error("error!");
      }
      // TODO: change this to 22 bit integer
      psum += (int)(bitActivations.bitInputValue[i] + bitWeights.bitInputValue[i]);
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
