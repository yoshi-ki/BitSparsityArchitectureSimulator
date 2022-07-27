#include "BFloatPEArray.h"
#include "BFloatPE.h"
#include "Utils.h"
#include "time.h"
#include <iostream>
#include <stdexcept> // std::runtime_error
#include "math.h"
#include <omp.h>


namespace simulator
{
  // bool that represents the PE Array is computing now or not

  // PE Array
  std::vector<std::vector<simulator::BFloatPE>> BFloatPEs(num_PE_height, std::vector<simulator::BFloatPE>(num_PE_width, BFloatPE()));

  // for unit test
  BFloatPEArray::BFloatPEArray(){};

  BFloatPEArray::BFloatPEArray(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputExpMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightExpMemories,
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

    BFloatPEArray::outputStatus = 0;
    BFloatPEArray::input_height = input_height;
    BFloatPEArray::input_width = input_width;
    BFloatPEArray::kernel_height = kernel_height;
    BFloatPEArray::kernel_width = kernel_height;
    BFloatPEArray::stride = stride;
    BFloatPEArray::num_output_channel = num_output_channel;

    BFloatPEArray::output_height = ((input_height - kernel_height) / stride) + 1;
    BFloatPEArray::output_width = ((input_width - kernel_width) / stride) + 1;

    outputMemory = std::vector<std::vector<std::vector<int>>>(num_output_channel, std::vector<std::vector<int>>(output_height, std::vector<int>(output_width)));
    outputExpMemory = std::vector<std::vector<std::vector<int>>>(num_output_channel, std::vector<std::vector<int>>(output_height, std::vector<int>(output_width)));

    inputValuesFifos = std::vector<std::vector<std::deque<FIFOValues>>>(num_PE_width, std::vector<std::deque<FIFOValues>>(num_PE_parallel, std::deque<FIFOValues>()));
    inputExpFifos = std::vector<std::vector<std::deque<int>>>(num_PE_width, std::vector<std::deque<int>>(num_PE_parallel, std::deque<int>()));
    weightValuesFifos = std::vector<std::vector<std::deque<FIFOValues>>> (num_PE_height, std::vector<std::deque<FIFOValues>>(num_PE_parallel, std::deque<FIFOValues>()));
    weightExpFifos = std::vector<std::vector<std::deque<int>>>(num_PE_height, std::vector<std::deque<int>>(num_PE_parallel, std::deque<int>()));

    // convert input activation and weight to the bit format,
    // because we only need bit format value
    convertInputMemoriesToFifos(inputMemories, inputExpMemories, inputValuesFifos, inputExpFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);
    convertWeightMemoriesToFifos(weightMemories, weightExpMemories, weightValuesFifos, weightExpFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);

    // Initialize states to control PEs
    inputControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_width);
    weightControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_height);

    preDecodedInputs = v<DecodedRegister>(num_PE_width, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_decodedRegister)), v<v<bool>>(num_PE_parallel, v<bool>(num_decodedRegister)), v<v<bool>>(num_PE_parallel, v<bool>(num_decodedRegister))});
    decodedInputs = v<DecodedRegister>(num_PE_width, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_decodedRegister)), v<v<bool>>(num_PE_parallel, v<bool>(num_decodedRegister)), v<v<bool>>(num_PE_parallel, v<bool>(num_decodedRegister))});
    decodedWeights = v<DecodedRegister>(num_PE_height, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_decodedRegister)), v<v<bool>>(num_PE_parallel, v<bool>(num_decodedRegister)), v<v<bool>>(num_PE_parallel, v<bool>(num_decodedRegister))});

    sharedExpForInputs = v<int>(num_PE_width);
    sharedExpForWeights = v<int>(num_PE_height);
    psumShiftedWidths = v<int>(num_PE_width);

    outputOfPEs = v<v<int>>(num_PE_height, std::vector<int>(num_PE_width));
    outputExpOfPEs = v<v<int>>(num_PE_height, std::vector<int>(num_PE_width));

    inputsForPEs = std::vector<PEInput>(num_PE_width, PEInput{v<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel)});
    weightsForPEs = std::vector<PEInput>(num_PE_height, PEInput{v<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel)});
  }

  bool BFloatPEArray::execute_one_step()
  {
    // if all of the value fifos become empty, we finish execution
    if(isLayerFinished(inputValuesFifos, weightValuesFifos)){
      busy = false;
      return busy;
    }


    // decode values (this circuit always run) (just look at the first element and do not change fifo)
    decodeValuesToBitsWithLeadingOne(inputValuesFifos, inputExpFifos, preDecodedInputs);
    decodeValuesToBitsWithLeadingOne(weightValuesFifos, weightExpFifos, decodedWeights);

    // exp extraction
    // do not change fifo itself. change decodedInputs.
    // extract psumShiftWidth and shift bit position in corresponsindg decodedInputs
    extractInputExpFromFifos(inputExpFifos, preDecodedInputs, decodedInputs, sharedExpForInputs, psumShiftedWidths);

    // etract weight exp
    extractWeightExpFromFifos(weightExpFifos, sharedExpForWeights);

    // we check the pe status and decide we send values or not and which to send for PEs.
    createInputForPEsBasedOnControllerStatus(decodedInputs, inputControllerStatusForPEs, inputsForPEs, num_PE_width);
    createInputForPEsBasedOnControllerStatus(decodedWeights, weightControllerStatusForPEs, weightsForPEs, num_PE_height);

    // #pragma omp parallel for
    for (int h = 0; h < num_PE_height; h++)
    {
      for (int w = 0; w < num_PE_width; w++){
        BFloatPEs[h][w].execute_one_step(inputsForPEs[w], weightsForPEs[h], psumShiftedWidths[w]);
      }
    }

    // update pe status
    updatePEStatus(inputControllerStatusForPEs, weightControllerStatusForPEs, inputValuesFifos, inputExpFifos, weightValuesFifos, weightExpFifos, decodedInputs, decodedWeights);

    bool finishedPsumExecution = isFinishedPSumExecution(inputControllerStatusForPEs, weightControllerStatusForPEs);

    // for mock of step 1, we write output if all PEs finish
    if (finishedPsumExecution)
    {
      // get output of PEs
      for (int h = 0; h < num_PE_height; h++)
      {
        for (int w = 0; w < num_PE_width; w++){
          std::pair<int,int> p = BFloatPEs[h][w].get_psum(sharedExpForInputs[w],sharedExpForWeights[h]);
          outputExpOfPEs[h][w] = p.first;
          outputOfPEs[h][w] = p.second;
        }
      }

      // TODO: something wired might happen when 0 input
      // write the output of PEs to the corresponding output position
      writeOutput(outputOfPEs, outputExpOfPEs, outputMemory, outputExpMemory, outputStatus, output_height, output_width, num_output_channel);
      outputStatus = outputStatus + 1;

      // update PE Status again to read next layers
      // finishedPSum -> false, isWaiting -> false, index -> 0, pop fifo
      updatePEStatusWhenPsumFinish(inputControllerStatusForPEs, weightControllerStatusForPEs, inputValuesFifos, inputExpFifos, weightValuesFifos, weightExpFifos);

      // reset state inside PE
      for (int h = 0; h < num_PE_height; h++){
        for (int w = 0; w < num_PE_width; w++){
          BFloatPEs[h][w].reset_state();
        }
      }

      // // reset decoded inputs and decoded weights
      // for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++){
      //   for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
      //     for (int i = 0; i < 8; i++){
      //       decodedInputs[inputFifoIndex].isValids[bitIndex][i] = false;
      //     }
      //   }
      // }
      // for (int weightFifoIndex = 0; weightFifoIndex < num_PE_width; weightFifoIndex++){
      //   for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
      //     for (int i = 0; i < 8; i++){
      //       decodedWeights[weightFifoIndex].isValids[bitIndex][i] = false;
      //     }
      //   }
      // }
    }
    return busy;
  };
}