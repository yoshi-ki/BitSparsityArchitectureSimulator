#include "PEArray.h"
#include "PE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

namespace simulator
{
  // bit fifos
  std::vector<std::queue<std::uint8_t>> bitInputFifos;
  std::vector<std::queue<std::uint8_t>> bitWeightFifos;

  // bool that represents the PE Array is computing now or not
  bool busy;

  // PE Array
  int num_PE_height = 8;
  int num_PE_width = 4;
  std::vector<std::vector<simulator::PE>> PEs(num_PE_height, std::vector<simulator::PE>(num_PE_width));

  PEArray::PEArray(
    std::vector<std::vector<std::int8_t>>& inputMemories,
    std::vector<std::vector<std::int8_t>>& weightMemories
  )
  {
    // convert input activation and weight to the bit format,
    // because we only need bit format value
    convertMemoriesToBitFifos(inputMemories, weightMemories, bitInputFifos, bitWeightFifos);
  }

  bool PEArray::execute_one_step()
  {
    // we use bit fifo for mock purpose
    std::vector<std::vector<int>> outputOfPEs(num_PE_height, std::vector<int>(num_PE_width));
    for (int h = 0; h < num_PE_height; h++){
      for (int w = 0; w < num_PE_width; w++){
        // (std::vector<unsigned int>& bitActivations, std::vector<unsigned int>& bitWeights)
        outputOfPEs[h][w] = PEs[h][w].execute_one_step();
      }
    }

    // pop all fifos here

    return busy;
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
      for (int i = 7; i >= 0; i--){
        int mask = 1 << i;
        if((val & mask) > 0){
          bitFifo.push((std::uint8_t)i);
        }
      }
    }
  }
}
