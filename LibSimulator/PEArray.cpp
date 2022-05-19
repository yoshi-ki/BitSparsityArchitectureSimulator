#include "PEArray.h"
#include "PE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

namespace simulator
{
  PEArray::PEArray(){
    // convert input activation and weight to the bit format,
    // because we only need bit format value
    convertMemoriesToBitFifos(inputMemories, weightMemories, bitInputFifos, bitWeightFifos);
  }

  int PEArray::execute_one_step(std::vector<unsigned int>& bitActivations, std::vector<unsigned int>& bitWeights)
  {
    for (int i = 0; i < bitActivations.size(); i++){
      if (bitActivations[i] >= 8 || bitWeights[i] >= 8){
        throw std::runtime_error("error!");
      }
      psum += (int)(bitActivations[i] + bitWeights[i]);
    }
    return psum;
  }

  int PEArray::get_psum()
  {
    return psum;
  }

  void PEArray::convertMemoriesToBitFifos(
    std::vector<std::vector<std::int8_t>> inputMemories,
    std::vector<std::vector<std::int8_t>> weightMemories,
    std::vector<std::queue<std::uint8_t>> bitInputFifos,
    std::vector<std::queue<std::uint8_t>> bitWeightFifos
  )
  {
    // by using the layer config, we will consume the values

    // intermediate values
    std::vector<std::queue<std::int8_t>> inputValuesFifos;
    std::vector<std::queue<std::int8_t>> weightValuesFifos;

    // convert to bit
    for (int i = 0; i < inputValuesFifos.size(); i++){
      convertValuesFifoToBitFifo(inputValuesFifos[i], bitInputFifos[i]);
    }
    for (int i = 0; i < weightValuesFifos.size(); i++){
      convertValuesFifoToBitFifo(weightValuesFifos[i], bitWeightFifos[i]);
    }
  }

  // private function to help converting
  void PEArray::convertValuesFifoToBitFifo(
    std::queue<std::int8_t> valueFifo,
    std::queue<std::uint8_t> bitFifo
  )
  {
    while(!valueFifo.empty()){
      int8_t val = valueFifo.front();
      valueFifo.pop();
      for (int i = 0; i < 8; i ++){
        int mask = 1 << i;
      }
    }
  }
}
