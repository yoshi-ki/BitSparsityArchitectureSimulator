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
  int num_PE_parallel = 16; // PE consumes 16 bits at one time

  PEArray::PEArray(
    std::vector<std::vector<std::int8_t>>& inputMemories,
    std::vector<std::vector<std::int8_t>>& weightMemories,
    int kernel_height,
    int kernel_width,
    int input_height,
    int input_width,
    int stride
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
    // intermediate values
    std::vector<std::vector<std::queue<std::int8_t>>> inputValuesFifos(num_PE_width, std::vector<std::queue<std::int8_t>>(num_PE_parallel));
    std::vector<std::vector<std::queue<std::int8_t>>> weightValuesFifos(num_PE_height, std::vector<std::queue<std::int8_t>>(num_PE_parallel));

    // by using the layer config, we will consume the values

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

  void PEArray::convertInputMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> inputMemories,
    std::vector<std::vector<std::queue<std::int8_t>>> inputValuesFifos,
    int input_channel_group,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride
  )
  {
    for (int memoryIndex = 0; memoryIndex < inputMemories.size(); memoryIndex++){
      // input is divided into four areas, so we need to compute the size of the each areas
      int partialInputHeight = (memoryIndex == 0 || memoryIndex == 3) ? input_height / 2 + input_height % 2 : input_height / 2;
      int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? input_width / 2 + input_width % 2 : input_width / 2;
      for (int channelGroup = 0; channelGroup < input_channel_group; channelGroup++){
        // start position of window
        for (int windowStartHeight = 0; windowStartHeight < partialInputHeight - kernel_height + 1; windowStartHeight++){
          for (int windowStartWidth = 0; windowStartWidth < partialInputWidth - kernel_width + 1; windowStartWidth++){
            for (int kh = 0; kh < kernel_height; kh++){
              for (int kw = 0; kw < kernel_width; kw++){
                // input
                for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                  inputValuesFifos[memoryIndex][input_channel].push(\
                    inputMemories[memoryIndex][channelGroup][windowStartHeight + kernel_height][windowStartWidth + kernel_width][input_channel]);
                }
              }
            }
          }
        }
      }
    }
  }

  void PEArray::convertWeightMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> weightMemories,
    std::vector<std::vector<std::queue<std::int8_t>>> weightValuesFifos,
    int input_channel_group,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      int memoryIndex = output_channel % 8;
      int partialInputHeight = (memoryIndex == 0 || memoryIndex == 3) ? input_height / 2 + input_height % 2 : input_height / 2;
      int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? input_width / 2 + input_width % 2 : input_width / 2;
      for (int channelGroup = 0; channelGroup < input_channel_group; channelGroup++){
        // start position of window
        for (int windowStartHeight = 0; windowStartHeight < partialInputHeight - kernel_height + 1; windowStartHeight++){
          for (int windowStartWidth = 0; windowStartWidth < partialInputWidth - kernel_width + 1; windowStartWidth++){
            for (int kh = 0; kh < kernel_height; kh++){
              for (int kw = 0; kw < kernel_width; kw++){
                // input
                for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                  weightValuesFifos[memoryIndex][input_channel].push(
                    weightMemories[memoryIndex][channelGroup][windowStartHeight + kernel_height][windowStartWidth + kernel_width][input_channel]);
                }
              }
            }
          }
        }
      }
    }
  }

}
