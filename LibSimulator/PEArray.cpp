#include "PEArray.h"
#include "PE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

namespace simulator
{
  // bool that represents the PE Array is computing now or not

  // PE Array
  int num_PE_height = 8;
  int num_PE_width = 4;
  std::vector<std::vector<simulator::PE>> PEs(num_PE_height, std::vector<simulator::PE>(num_PE_width));
  int num_PE_parallel = 16; // PE consumes 16 bits at one time

  PEArray::PEArray(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& weightMemories,
    int num_input_channel_group,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    busy = true;

    inputValuesFifos = std::vector<std::vector<std::queue<std::int8_t>>>(num_PE_width, std::vector<std::queue<std::int8_t>>(num_PE_parallel));
    weightValuesFifos = std::vector<std::vector<std::queue<std::int8_t>>> (num_PE_height, std::vector<std::queue<std::int8_t>>(num_PE_parallel));

    // convert input activation and weight to the bit format,
    // because we only need bit format value
    convertInputMemoriesToFifos(inputMemories, inputValuesFifos, num_input_channel_group, input_height, input_width, kernel_height, kernel_width, stride);
    convertWeightMemoriesToFifos(weightMemories, weightValuesFifos, num_input_channel_group, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);

    // states to control PEs
    inputControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_width);
    weightControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_height);

    // Fifos that correspond to each PEs
    bitInputs = std::vector<std::vector<std::vector<std::uint8_t>>>(num_PE_width, std::vector<std::vector<std::uint8_t>>(num_PE_parallel));
    bitWeights = std::vector<std::vector<std::vector<std::uint8_t>>>(num_PE_height, std::vector<std::vector<std::uint8_t>>(num_PE_parallel));
  }

  bool PEArray::execute_one_step()
  {
    // if all of the value fifos become empty, we finish execution
    if(isLayerFinished(inputValuesFifos, weightValuesFifos)){
      busy = false;
      return busy;
    }

    // decode values (this circuit always run) (just look at the first element and do not change fifo)
    decodeValuesToBits(inputValuesFifos, bitInputs);
    decodeValuesToBits(weightValuesFifos, bitWeights);

    // based on the PE controller status, prepare input for PEs
    auto inputsForPEs = std::vector<std::vector<unsigned int>>(num_PE_width, std::vector<unsigned int>(num_PE_parallel));
    auto weightsForPEs = std::vector<std::vector<unsigned int>>(num_PE_height, std::vector<unsigned int>(num_PE_parallel));
    createInputForPEsBasedOnControllerStatus(bitInputs, inputControllerStatusForPEs, inputsForPEs);
    createInputForPEsBasedOnControllerStatus(bitWeights, weightControllerStatusForPEs, weightsForPEs);

    // we use
    std::vector<std::vector<outputPE>> outputOfPEs(num_PE_height, std::vector<outputPE>(num_PE_width));
    for (int h = 0; h < num_PE_height; h++){
      for (int w = 0; w < num_PE_width; w++){
        outputOfPEs[h][w] = PEs[h][w].execute_one_step(inputsForPEs[w], weightsForPEs[h]);
      }
    }

    // update pe status
    // TODO: need to add layer psum end information
    updatePEStatus(inputControllerStatusForPEs, weightControllerStatusForPEs, inputValuesFifos, weightValuesFifos, bitInputs, bitWeights);

    // for mock of step 1, we write output if all PEs finish
    if (finishedpsumExecution)
    {
      // write to output of PE to the corresponding output position

      // reset state inside PE
      for (int h = 0; h < num_PE_height; h++){
        for (int w = 0; w < num_PE_width; w++){
          PEs[h][w].reset_state();
        }
      }
    }
  }

  bool PEArray::isLayerFinished(
    std::vector<std::vector<std::queue<std::int8_t>>>& inputFifos,
    std::vector<std::vector<std::queue<std::int8_t>>>& weightFifos
  )
  {
    bool isFinished = true;
    for (int i = 0; i < inputFifos.size(); i++){
      for (int j = 0; j < inputFifos[i].size(); j++){
        isFinished = isFinished && inputFifos[i][j].empty();
      }
    }
    for (int i = 0; i < weightFifos.size(); i++){
      for (int j = 0; j < weightFifos[i].size(); j++){
        isFinished = isFinished && weightFifos[i][j].empty();
      }
    }
    return isFinished;
  }

  void PEArray::decodeValuesToBits(
    std::vector<std::vector<std::queue<std::int8_t>>>& valueFifos,
    std::vector<std::vector<std::vector<std::uint8_t>>>bitRepresentations
  )
  {
    for (int fifoIndex = 0; fifoIndex < valueFifos.size(); fifoIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        int8_t val = valueFifos[fifoIndex][input_channel].front();
        int bitVectorIndex = 0;
        for (int i = 7; i >= 0; i--)
        {
          int mask = 1 << i;
          if ((val & mask) > 0)
          {
            bitRepresentations[fifoIndex][input_channel][bitVectorIndex] = (std::uint8_t)i;
            bitVectorIndex++;
          }
        }
      }
    }
  }

  void PEArray::createInputForPEsBasedOnControllerStatus(
    std::vector<std::vector<std::vector<std::uint8_t>>>& bitRepresentations,
    std::vector<PEControllerStatus>& controllerStatusForPEs,
    std::vector<std::vector<unsigned int>>& representationsForPEs
  )
  {
    for (int fifoIndex = 0; fifoIndex < num_PE_width; fifoIndex++){
      auto controllerStatus = controllerStatusForPEs[fifoIndex];
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        if(controllerStatus.isWaiting[bitIndex]){
          // when waiting for next value
          representationsForPEs[fifoIndex][bitIndex] = 0;
        }
        else{
          representationsForPEs[fifoIndex][bitIndex] =
              (unsigned int)bitRepresentations[fifoIndex][bitIndex][controllerStatus.nextProcessIndex[bitIndex]];
        }
      }
    }
  };

  void PEArray::updatePEStatus(
    std::vector<PEControllerStatus>& inputControllerStatusForPEs,
    std::vector<PEControllerStatus>& weightControllerStatusForPEs,
    std::vector<std::vector<std::queue<std::int8_t>>>& inputValuesFifos,
    std::vector<std::vector<std::queue<std::int8_t>>>& weightValuesFifos,
    std::vector<std::vector<std::vector<std::uint8_t>>>& bitInputs,
    std::vector<std::vector<std::vector<std::uint8_t>>>& bitWeights
  )
  {
    auto newInputControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_width);
    auto newWeightControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_height);

    // create new input controller status
    for (int fifoIndex = 0; fifoIndex < num_PE_width; fifoIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        int nextProcessIndex = inputControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] + 1;
        newInputControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] = nextProcessIndex;
        newInputControllerStatusForPEs[fifoIndex].isWaiting[bitIndex] = nextProcessIndex < bitInputs[fifoIndex][bitIndex].size();
      }
    }

    // create new weight controller status
    for (int fifoIndex = 0; fifoIndex < num_PE_height; fifoIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        bool weightForThisBitNext = true;
        for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++)
        {
          weightForThisBitNext = weightForThisBitNext && newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex];
        }
        if (weightForThisBitNext){
          // we will consume weight next bit from the next cycle
          int nextProcessIndex = weightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] + 1;
          newWeightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] = nextProcessIndex;

          bool weightFifoWaiting = nextProcessIndex >= bitWeights[fifoIndex][bitIndex].size();
          newWeightControllerStatusForPEs[fifoIndex].isWaiting[bitIndex] = weightFifoWaiting;

          // we need to update the status for input controller if new weight bit is produced
          if (!weightFifoWaiting){
            for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++)
            {
              newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = false;
              newInputControllerStatusForPEs[inputFifoIndex].nextProcessIndex[bitIndex] = 0;
            }
          }
        }
      }
    }

    // decide we will update the fifo or not
    bool finishedLayerExecution = false;
    for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
      bool updateFifo = true;
      for (int weightFifoIndex = 0; weightFifoIndex < num_PE_height; weightFifoIndex++)
      {
        updateFifo = updateFifo && newWeightControllerStatusForPEs[weightFifoIndex].isWaiting[bitIndex];
      }

      if(updateFifo){
        for (int weightFifoIndex = 0; weightFifoIndex < num_PE_height; weightFifoIndex++)
        {
          if (!weightValuesFifos[weightFifoIndex][bitIndex].empty()){
            weightValuesFifos[weightFifoIndex][bitIndex].pop();
            newWeightControllerStatusForPEs[weightFifoIndex].isWaiting[bitIndex] = false;
            newWeightControllerStatusForPEs[weightFifoIndex].nextProcessIndex[bitIndex] = 0;
          }
        }
        for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++){
          if (!inputValuesFifos[inputFifoIndex][bitIndex].empty()){
            inputValuesFifos[inputFifoIndex][bitIndex].pop();
            newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = false;
            newInputControllerStatusForPEs[inputFifoIndex].nextProcessIndex[bitIndex] = 0;
          }
        }
      }
    }

    inputControllerStatusForPEs = newInputControllerStatusForPEs;
    weightControllerStatusForPEs = newWeightControllerStatusForPEs;

    return ;
  };

  void PEArray::convertMemoriesToBitFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& weightMemories,
    std::vector<std::vector<std::queue<std::uint8_t>>>& bitInputFifos,
    std::vector<std::vector<std::queue<std::uint8_t>>>& bitWeightFifos,
    int num_input_channel_group,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    // intermediate values
    std::vector<std::vector<std::queue<std::int8_t>>> inputValuesFifos(num_PE_width, std::vector<std::queue<std::int8_t>>(num_PE_parallel));
    std::vector<std::vector<std::queue<std::int8_t>>> weightValuesFifos(num_PE_height, std::vector<std::queue<std::int8_t>>(num_PE_parallel));

    // by using the layer config, we will consume the values
    convertInputMemoriesToFifos(inputMemories, inputValuesFifos, num_input_channel_group, input_height, input_width, kernel_height, kernel_width, stride);
    convertWeightMemoriesToFifos(weightMemories, weightValuesFifos, num_input_channel_group, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);

    // convert to bit
    convertValuesFifosToBitFifos(inputValuesFifos, bitInputFifos);
    convertValuesFifosToBitFifos(weightValuesFifos, bitWeightFifos);
  }

  // private function to help converting
  void PEArray::convertValuesFifosToBitFifos(
    std::vector<std::vector<std::queue<std::int8_t>>>& valuesFifos,
    std::vector<std::vector<std::queue<std::uint8_t>>>& bitFifos
  )
  {
    for (int fifoIndex = 0; fifoIndex < valuesFifos.size(); fifoIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        while (!valuesFifos.empty())
        {
          int8_t val = valuesFifos[fifoIndex][input_channel].front();
          valuesFifos[fifoIndex][input_channel].pop();
          for (int i = 7; i >= 0; i--)
          {
            int mask = 1 << i;
            if ((val & mask) > 0)
            {
              bitFifos[fifoIndex][input_channel].push((std::uint8_t)i);
            }
          }
        }
      }
    }
  }

  void PEArray::convertInputMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
    std::vector<std::vector<std::queue<std::int8_t>>>& inputValuesFifos,
    int num_input_channel_group,
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
      for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
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
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& weightMemories,
    std::vector<std::vector<std::queue<std::int8_t>>>& weightValuesFifos,
    int num_input_channel_group,
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
      for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
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
