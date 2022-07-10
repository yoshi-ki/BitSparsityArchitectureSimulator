#pragma once
#include <vector>
#include <deque>

template<class T>
using v = std::vector<T>;

namespace simulator
{
  // const int num_PE_height = 8;
  // const int num_PE_width = 4;
  // const int num_PE_parallel = 16; // PE consumes 16 bits at once
  // const int num_bit_size = 8;

  // struct PEControllerStatus{
  //   std::vector<bool> isWaiting = std::vector<bool>(num_PE_parallel);
  //   std::vector<int> nextProcessIndex = std::vector<int>(num_PE_parallel);
  //   std::vector<bool> finishedPSum = std::vector<bool>(num_PE_parallel);
  // };


  // struct DecodedRegister{
  //   // it is for 16 inputs and values just decoded
  //   std::vector<std::vector<unsigned int>> bitInputValues; // [inputChannel][bitIndex]
  //   std::vector<std::vector<bool>> isNegatives;
  //   std::vector<std::vector<bool>> isValids;
  // };

  // struct PEInput{
  //   // it is for 16 inputs
  //   std::vector<unsigned int> bitInputValue;
  //   std::vector<bool> isNegative;
  //   std::vector<bool> isValid;
  // };

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
      int outputStatus;

      // value fifos
      std::vector<std::vector<std::deque<FIFOValues>>> inputValuesFifos;
      std::vector<std::vector<std::deque<int>>> inputExpFifos;
      std::vector<std::vector<std::deque<FIFOValues>>> weightValuesFifos;
      std::vector<std::vector<std::deque<int>>> weightExpFifos;

      // bit fifos
      std::vector<DecodedRegister> decodedInputs;
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
