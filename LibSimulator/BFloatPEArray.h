#pragma once
#include <vector>
#include <deque>
#include "Utils.h"

template<class T>
using v = std::vector<T>;

namespace simulator
{
  class BFloatPEArray
  {
    public:
      /* initializer */
      // what initializer does is to set the input and weight values, and layer config
      BFloatPEArray();

      BFloatPEArray(
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
      );

      /* method */
      bool execute_one_step();

      /* member */

      // input and output memory
      std::vector<std::vector<std::vector<int>>> outputMemory;
      std::vector<std::vector<std::vector<int>>> outputExpMemory;
      int outputStatus;

      // value fifos
      std::vector<std::vector<std::deque<FIFOValues>>> inputValuesFifos;
      std::vector<std::vector<std::deque<int>>> inputExpFifos;
      std::vector<std::vector<std::deque<FIFOValues>>> weightValuesFifos;
      std::vector<std::vector<std::deque<int>>> weightExpFifos;

      // bit fifos
      std::vector<DecodedRegister> decodedInputs;
      std::vector<DecodedRegister> preDecodedInputs;
      std::vector<DecodedRegister> decodedWeights;

      // output of PEs
      std::vector<std::vector<int>> outputOfPEs;
      std::vector<std::vector<int>> outputExpOfPEs;

      std::vector<PEInput> inputsForPEs;
      std::vector<PEInput> weightsForPEs;

      // states to control PEs
      std::vector<PEControllerStatus> inputControllerStatusForPEs;
      std::vector<PEControllerStatus> weightControllerStatusForPEs;

      // busy or not, by this information, memory decides whether it sends data or not
      bool busy;

      v<int> sharedExpForInputs;
      v<int> sharedExpForWeights;
      v<int> psumShiftedWidths;

      int num_input_channel_group;
      int input_height;
      int input_width;
      int kernel_height;
      int kernel_width;
      int stride;
      int num_output_channel;
      int output_height;
      int output_width;
  };
}
