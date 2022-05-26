#include "PEArray.h"
#include "PE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

namespace simulator
{
  // bool that represents the PE Array is computing now or not

  // PE Array
  std::vector<std::vector<simulator::PE>> PEs(num_PE_height, std::vector<simulator::PE>(num_PE_width));

  // for unit test
  PEArray::PEArray(){};

  PEArray::PEArray(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
    int num_input_channel,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    busy = true;

    PEArray::outputStatus = 0;
    PEArray::input_height = input_height;
    PEArray::input_width = input_width;
    PEArray::kernel_height = kernel_height;
    PEArray::kernel_width = kernel_height;
    PEArray::stride = stride;
    PEArray::num_output_channel = num_output_channel;

    PEArray::output_height = ((input_height - kernel_height) / stride) + 1;
    PEArray::output_width = ((input_width - kernel_width) / stride) + 1;

    outputMemory = std::vector<std::vector<std::vector<int>>>(num_output_channel, std::vector<std::vector<int>>(output_height, std::vector<int>(output_width)));

    inputValuesFifos = std::vector<std::vector<std::deque<FIFOValues>>>(num_PE_width, std::vector<std::deque<FIFOValues>>(num_PE_parallel));
    weightValuesFifos = std::vector<std::vector<std::deque<FIFOValues>>> (num_PE_height, std::vector<std::deque<FIFOValues>>(num_PE_parallel));

    // convert input activation and weight to the bit format,
    // because we only need bit format value
    convertInputMemoriesToFifos(inputMemories, inputValuesFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride);
    convertWeightMemoriesToFifos(weightMemories, weightValuesFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);

    // Initialize states to control PEs
    inputControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_width);
    weightControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_height);

    // Initialize Fifos that correspond to each PEs
    bitInputs = v<v<v<unsigned int>>>(num_PE_width, v<v<unsigned int>>(num_PE_parallel, v<unsigned int>(num_bit_size)));
    bitWeights = v<v<v<unsigned int>>>(num_PE_height, v<v<unsigned int>>(num_PE_parallel, v<unsigned int>(num_bit_size)));
  }

  bool PEArray::execute_one_step()
  {
    // if all of the value fifos become empty, we finish execution
    if(isLayerFinished(inputValuesFifos, weightValuesFifos)){
      // output for debug
      // for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      //   for (int output_heigh = 0; output_heigh < output_height; output_heigh++){
      //     for (int output_widt = 0; output_widt < output_width; output_widt++){
      //       std::cout << outputMemory[output_channel][output_heigh][output_widt] <<std::endl ;
      //     }
      //   }
      // }
      busy = false;
      return busy;
    }

    // decode values (this circuit always run) (just look at the first element and do not change fifo)
    decodeValuesToBits(inputValuesFifos, bitInputs);
    decodeValuesToBits(weightValuesFifos, bitWeights);

    // based on the PE controller status, prepare input for PEs
    auto inputsForPEs = std::vector<std::vector<unsigned int>>(num_PE_width, std::vector<unsigned int>(num_PE_parallel));
    auto weightsForPEs = std::vector<std::vector<unsigned int>>(num_PE_height, std::vector<unsigned int>(num_PE_parallel));
    createInputForPEsBasedOnControllerStatus(bitInputs, inputControllerStatusForPEs, inputsForPEs, num_PE_width);
    createInputForPEsBasedOnControllerStatus(bitWeights, weightControllerStatusForPEs, weightsForPEs, num_PE_height);

    // we use
    std::vector<std::vector<int>> outputOfPEs(num_PE_height, std::vector<int>(num_PE_width));
    for (int h = 0; h < num_PE_height; h++){
      for (int w = 0; w < num_PE_width; w++){
        outputOfPEs[h][w] = PEs[h][w].execute_one_step(inputsForPEs[w], weightsForPEs[h]);
      }
    }

    // update pe status
    updatePEStatus(inputControllerStatusForPEs, weightControllerStatusForPEs, inputValuesFifos, weightValuesFifos, bitInputs, bitWeights);

    bool finishedPsumExecution = isFinishedPSumExecution(inputControllerStatusForPEs, weightControllerStatusForPEs);

    // for mock of step 1, we write output if all PEs finish
    if (finishedPsumExecution)
    {
      // TODO: something wired might happen when 0 input
      // write the output of PEs to the corresponding output position
      writeOutput(outputOfPEs, outputMemory, outputStatus, output_height, output_width, num_output_channel);
      outputStatus = outputStatus + 1;

      // update PE Status again to read next layers
      // finishedPSum -> false, isWaiting -> false, index -> 0, pop fifo
      updatePEStatusWhenPsumFinish(inputControllerStatusForPEs, weightControllerStatusForPEs, inputValuesFifos, weightValuesFifos);

      // reset state inside PE
      for (int h = 0; h < num_PE_height; h++){
        for (int w = 0; w < num_PE_width; w++){
          PEs[h][w].reset_state();
        }
      }
    }
    return busy;
  };

  bool PEArray::isLayerFinished(
    std::vector<std::vector<std::deque<FIFOValues>>>& inputFifos,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightFifos
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
  };

  bool PEArray::isFinishedPSumExecution(
    std::vector<PEControllerStatus> inputControllerStatusForPEs,
    std::vector<PEControllerStatus> weightControllerStatusForPEs
  )
  {
    bool isFinished = true;
    for (int i = 0; i < inputControllerStatusForPEs.size(); i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        isFinished = isFinished && inputControllerStatusForPEs[i].finishedPSum[bitIndex];
      }
    }
    for (int i = 0; i < weightControllerStatusForPEs.size(); i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        isFinished = isFinished && weightControllerStatusForPEs[i].finishedPSum[bitIndex];
      }
    }
    return isFinished;
  };

  void PEArray::decodeValuesToBits(
    std::vector<std::vector<std::deque<FIFOValues>>>& valueFifos,
    std::vector<std::vector<std::vector<unsigned int>>>bitRepresentations
  )
  {
    for (int fifoIndex = 0; fifoIndex < valueFifos.size(); fifoIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        int8_t val = valueFifos[fifoIndex][input_channel].front().value;
        int bitVectorIndex = 0;
        for (int i = 7; i >= 0; i--)
        {
          int mask = 1 << i;
          if ((val & mask) > 0)
          {
            bitRepresentations[fifoIndex][input_channel][bitVectorIndex] = (unsigned int)i;
            bitVectorIndex++;
          }
        }
      }
    }
  };

  void PEArray::createInputForPEsBasedOnControllerStatus(
    std::vector<std::vector<std::vector<unsigned int>>>& bitRepresentations,
    std::vector<PEControllerStatus>& controllerStatusForPEs,
    std::vector<std::vector<unsigned int>>& representationsForPEs,
    int num_Fifo
  )
  {
    for (int fifoIndex = 0; fifoIndex < num_Fifo; fifoIndex++){
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
    std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
    std::vector<std::vector<std::vector<unsigned int>>>& bitInputs,
    std::vector<std::vector<std::vector<unsigned int>>>& bitWeights
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

      // update for activation and weight is done at the same time
      if(updateFifo){
        for (int weightFifoIndex = 0; weightFifoIndex < num_PE_height; weightFifoIndex++)
        {
          if (!weightValuesFifos[weightFifoIndex][bitIndex].empty()){
            // if the value is the last values of psum, we need to set wait status
            if (weightControllerStatusForPEs[weightFifoIndex].finishedPSum[bitIndex] ||
                weightValuesFifos[weightFifoIndex][bitIndex].front().isLast)
            {
              newWeightControllerStatusForPEs[weightFifoIndex].finishedPSum[bitIndex] = true;
            }
            else{
              weightValuesFifos[weightFifoIndex][bitIndex].pop_front();
              newWeightControllerStatusForPEs[weightFifoIndex].isWaiting[bitIndex] = false;
              newWeightControllerStatusForPEs[weightFifoIndex].nextProcessIndex[bitIndex] = 0;
            }
          }
        }
        for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++){
          if (!inputValuesFifos[inputFifoIndex][bitIndex].empty()){
            // if the value is the last values of psum, we need to set wait status
            if (inputControllerStatusForPEs[inputFifoIndex].finishedPSum[bitIndex] ||
                inputValuesFifos[inputFifoIndex][bitIndex].front().isLast)
            {
              newInputControllerStatusForPEs[inputFifoIndex].finishedPSum[bitIndex] = true;
            }
            else{
              inputValuesFifos[inputFifoIndex][bitIndex].pop_front();
              newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = false;
              newInputControllerStatusForPEs[inputFifoIndex].nextProcessIndex[bitIndex] = 0;
            }
          }
        }
      }
    }

    inputControllerStatusForPEs = newInputControllerStatusForPEs;
    weightControllerStatusForPEs = newWeightControllerStatusForPEs;

    return ;
  };

  void PEArray::updatePEStatusWhenPsumFinish(
    std::vector<PEControllerStatus> & inputControllerStatusForPEs,
    std::vector<PEControllerStatus> & weightControllerStatusForPEs,
    std::vector<std::vector<std::deque<FIFOValues>>> & inputValuesFifos,
    std::vector<std::vector<std::deque<FIFOValues>>> & weightValuesFifos
  )
  {
    // finishedPSum -> false, isWaiting -> false, index -> 0, pop fifo
    for (int i = 0; i < inputControllerStatusForPEs.size(); i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        inputControllerStatusForPEs[i].finishedPSum[bitIndex] = false;
        inputControllerStatusForPEs[i].isWaiting[bitIndex] = false;
        inputControllerStatusForPEs[i].nextProcessIndex[bitIndex] = 0;
      }
    }
    for (int memoryIndex = 0; memoryIndex < inputValuesFifos.size(); memoryIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        inputValuesFifos[memoryIndex][input_channel].pop_front();
      }
    }

    for (int i = 0; i < weightControllerStatusForPEs.size(); i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        weightControllerStatusForPEs[i].finishedPSum[bitIndex] = false;
        weightControllerStatusForPEs[i].isWaiting[bitIndex] = false;
        weightControllerStatusForPEs[i].nextProcessIndex[bitIndex] = 0;
      }
    }
    for (int memoryIndex = 0; memoryIndex < inputValuesFifos.size(); memoryIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        weightValuesFifos[memoryIndex][input_channel].pop_front();
      }
    }
  };

  void PEArray::convertInputMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
    int num_input_channel,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride
  )
  {
    int num_input_channel_group = num_input_channel / num_PE_parallel + fmin(num_input_channel % num_PE_parallel, 1);
    for (int memoryIndex = 0; memoryIndex < inputMemories.size(); memoryIndex++){
      // input is divided into four areas, so we need to compute the size of the each areas
      int partialInputHeight = (memoryIndex == 0 || memoryIndex == 3) ? input_height / 2 + input_height % 2 : input_height / 2;
      int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? input_width / 2 + input_width % 2 : input_width / 2;
      // start position of window
      for (int windowStartHeight = 0; windowStartHeight < partialInputHeight - kernel_height + 1; windowStartHeight++){
        for (int windowStartWidth = 0; windowStartWidth < partialInputWidth - kernel_width + 1; windowStartWidth++){
          for (int kh = 0; kh < kernel_height; kh++){
            for (int kw = 0; kw < kernel_width; kw++){
              for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
                // input
                for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                  inputValuesFifos[memoryIndex][input_channel].push_front(
                      FIFOValues{
                          inputMemories[memoryIndex][windowStartHeight + kernel_height][windowStartWidth + kernel_width][channelGroup][input_channel],
                          false
                      });
                }
              }
            }
          }
        }
      }

      // insert end of the psum
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        if(inputValuesFifos[memoryIndex][input_channel].empty()){
          continue;
        }

        auto lastVal = inputValuesFifos[memoryIndex][input_channel].back();
        lastVal.value = true;
        inputValuesFifos[memoryIndex][input_channel].pop_back();
        inputValuesFifos[memoryIndex][input_channel].push_back(lastVal);
      }
    }
  }

  void PEArray::convertWeightMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
    int num_input_channel,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    int num_input_channel_group = num_input_channel / num_PE_parallel + fmin(num_input_channel % num_PE_parallel, 1);
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      int memoryIndex = output_channel % num_PE_height;
      int partialInputHeight = (memoryIndex == 0 || memoryIndex == 3) ? input_height / 2 + input_height % 2 : input_height / 2;
      int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? input_width / 2 + input_width % 2 : input_width / 2;

      // this is not the start position here, it is just iteration
      for (int windowStartHeight = 0; windowStartHeight < partialInputHeight - kernel_height + 1; windowStartHeight = windowStartHeight + stride){
        for (int windowStartWidth = 0; windowStartWidth < partialInputWidth - kernel_width + 1; windowStartWidth = windowStartWidth + stride){

          for (int kh = 0; kh < kernel_height; kh++){
            for (int kw = 0; kw < kernel_width; kw++){
              for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
                // input
                for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                  weightValuesFifos[memoryIndex][input_channel].push_front(
                    FIFOValues{
                        weightMemories[memoryIndex][output_channel / num_PE_height][kh][kw][channelGroup][input_channel],
                        false
                    });
                }
              }
            }
          }
        }
      }

      // insert end of the psum
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        if(weightValuesFifos[memoryIndex][input_channel].empty()){
          continue;
        }
        auto lastVal = weightValuesFifos[memoryIndex][input_channel].back();
        lastVal.value = true;
        weightValuesFifos[memoryIndex][input_channel].pop_back();
        weightValuesFifos[memoryIndex][input_channel].push_back(lastVal);
      }
    }
  };

  void PEArray::writeOutput(
    std::vector<std::vector<int>>& outputOfPEs,
    std::vector<std::vector<std::vector<int>>>& outputMemory,
    int outputStatus,
    int output_height,
    int output_width,
    int num_output_channel
  )
  {
    for (int h = 0; h < num_PE_height; h++)
    {
      for (int w = 0; w < num_PE_width; w++){
        int outputChannelGroup = outputStatus / (output_height * output_width);
        int outputPositionIndex = outputStatus % (output_height * output_width);
        int writeOutputChannel = h + 4 * outputChannelGroup;
        int writeOutputHeight = outputPositionIndex / output_width;
        int writeOutputWidth = outputPositionIndex % output_width;
        outputMemory[writeOutputChannel][writeOutputHeight][writeOutputWidth] = outputOfPEs[h][w];
      }
    }
  };

}