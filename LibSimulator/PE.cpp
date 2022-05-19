#include "PE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

namespace simulator
{
  PE::PE(){
    psum = 0;
  }

  int PE::execute_one_step(std::vector<unsigned int>& bitActivations, std::vector<unsigned int>& bitWeights)
  {
    for (int i = 0; i < bitActivations.size(); i++){
      if (bitActivations[i] >= 8 || bitWeights[i] >= 8){
        throw std::runtime_error("error!");
      }
      // TODO: change this to 22 bit integer
      psum += (int)(bitActivations[i] + bitWeights[i]);
    }
    return psum;
  }

  int PE::get_psum()
  {
    return psum;
  }
}
