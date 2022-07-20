#include "PEArray.h"
#include "PE.h"
#include "time.h"
#include "Utils.h"
#include <iostream>
#include <stdexcept> // std::runtime_error
#include "math.h"
#include <omp.h>


namespace simulator
{
  // bool that represents the PE Array is computing now or not

  // PE Array
  std::vector<std::vector<simulator::PE>> PEs(num_PE_height, std::vector<simulator::PE>(num_PE_width, PE()));

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

    inputValuesFifos = std::vector<std::vector<std::deque<FIFOValues>>>(num_PE_width, std::vector<std::deque<FIFOValues>>(num_PE_parallel, std::deque<FIFOValues>()));
    weightValuesFifos = std::vector<std::vector<std::deque<FIFOValues>>> (num_PE_height, std::vector<std::deque<FIFOValues>>(num_PE_parallel, std::deque<FIFOValues>()));

    // convert input activation and weight to the bit format,
    // because we only need bit format value
    convertInputMemoriesToFifos(inputMemories, inputValuesFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);
    convertWeightMemoriesToFifos(weightMemories, weightValuesFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);

    // Initialize states to control PEs
    inputControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_width);
    weightControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_height);

    decodedInputs = v<DecodedRegister>(num_PE_width, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_PE_parallel)), v<v<bool>>(num_PE_parallel, v<bool>(8)), v<v<bool>>(num_PE_parallel, v<bool>(8))});
    decodedWeights = v<DecodedRegister>(num_PE_height, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_PE_parallel)), v<v<bool>>(num_PE_parallel, v<bool>(8)), v<v<bool>>(num_PE_parallel, v<bool>(8))});

    outputOfPEs = v<v<int>>(num_PE_height, std::vector<int>(num_PE_width));

    inputsForPEs = std::vector<PEInput>(num_PE_width, PEInput{v<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel)});
    weightsForPEs = std::vector<PEInput>(num_PE_height, PEInput{v<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel)});
  }

  bool PEArray::execute_one_step()
  {
    // if all of the value fifos become empty, we finish execution
    if(isLayerFinished(inputValuesFifos, weightValuesFifos)){
      busy = false;
      return busy;
    }

    // decode values (this circuit always run) (just look at the first element and do not change fifo)
    decodeValuesToBits(inputValuesFifos, decodedInputs);
    decodeValuesToBits(weightValuesFifos, decodedWeights);

    // we check the pe status and decide we send values or not and which to send for PEs.
    createInputForPEsBasedOnControllerStatus(decodedInputs, inputControllerStatusForPEs, inputsForPEs, num_PE_width);
    createInputForPEsBasedOnControllerStatus(decodedWeights, weightControllerStatusForPEs, weightsForPEs, num_PE_height);

    // #pragma omp parallel for
    for (int h = 0; h < num_PE_height; h++)
    {
      for (int w = 0; w < num_PE_width; w++){
        outputOfPEs[h][w] = PEs[h][w].execute_one_step(inputsForPEs[w], weightsForPEs[h]);
      }
    }

    // update pe status
    updatePEStatus(inputControllerStatusForPEs, weightControllerStatusForPEs, inputValuesFifos, weightValuesFifos, decodedInputs, decodedWeights);

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

}