#include "PE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

namespace simulator
{
  PE::PE(){
    psum = 0;
    state = 0;
    endComputation = false;
  }

  outputPE PE::execute_one_step(
    std::vector<unsigned int>& bitActivations,
    std::vector<unsigned int>& bitWeights,
    std::vector<bool>& activationEnd,
    std::vector<bool>& weightEnd
  )
  {
    // there are two states of the PE,
    // one state is executing sum
    // another state is waiting for the sum to be absorbed

    // state: executing sum
    if(state == 0){
      for (int i = 0; i < bitActivations.size(); i++){
        if (bitActivations[i] >= 8 || bitWeights[i] >= 8){
          throw std::runtime_error("error!");
        }
        // TODO: change this to 22 bit integer
        psum += (int)(bitActivations[i] + bitWeights[i]);
      }
    }

    // state transition
    if (std::all_of(activationEnd.begin(), activationEnd.end(), [](bool x) { return x; } ) &&
        std::all_of(weightEnd.begin(), weightEnd.end(), [](bool x) { return x; } ))
    {
      state = 1;
      endComputation = true;
    }
    return outputPE{psum, endComputation};
  }

  void PE::reset_state(){
    state = 0;
    psum = 0;
    endComputation = 0;
  }

  int PE::get_psum()
  {
    return psum;
  }
}
