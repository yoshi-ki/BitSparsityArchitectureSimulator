#pragma once
#include <vector>
#include <deque>

template<class T>
using v = std::vector<T>;

namespace simulator
{
  const int num_PE_height = 8;
  const int num_PE_width = 4;
  const int num_PE_parallel = 16; // PE consumes 16 bits at once
  const int num_bit_size = 8;

  struct PEControllerStatus{
    std::vector<bool> isWaiting = std::vector<bool>(num_PE_parallel);
    std::vector<int> nextProcessIndex = std::vector<int>(num_PE_parallel);
    std::vector<bool> finishedPSum = std::vector<bool>(num_PE_parallel);
  };

  struct FIFOValues{
    int value;
    bool isLast;
  };

  struct DecodedRegister{
    // it is for 16 inputs and values just decoded
    std::vector<std::vector<unsigned int>> bitInputValues; // [inputChannel][bitIndex]
    std::vector<std::vector<bool>> isNegatives;
    std::vector<std::vector<bool>> isValids;
  };

  struct PEInput{
    // it is for 16 inputs
    std::vector<unsigned int> bitInputValue;
    std::vector<bool> isNegative;
    std::vector<bool> isValid;
  };

  class PEArray
  {
    public:
      /* initializer */
      // what initializer does is to set the input and weight values, and layer config
      PEArray();

      PEArray(
        std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
        std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
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
      // std::vector<std::vector<int>> inputMemories;
      // std::vector<std::vector<int>> weightMemories;
      std::vector<std::vector<std::vector<int>>> outputMemory;
      int outputStatus;

      // value fifos
      std::vector<std::vector<std::deque<FIFOValues>>> inputValuesFifos;
      std::vector<std::vector<std::deque<FIFOValues>>> weightValuesFifos;

      // bit fifos
      std::vector<DecodedRegister> decodedInputs;
      std::vector<DecodedRegister> decodedWeights;

      // states to control PEs
      // std::vector<bool> isInputFifoWaiting;
      // std::vector<bool> isWeightFifoWaiting;
      // std::vector<int> indexInputBit;
      // std::vector<int> indexWeightBit;
      std::vector<PEControllerStatus> inputControllerStatusForPEs;
      std::vector<PEControllerStatus> weightControllerStatusForPEs;

      // busy or not, by this information, memory decides whether it sends data or not
      bool busy;

    // private:
      void convertInputMemoriesToFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
        std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
        int num_input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride,
        int num_output_channel
      );

      void convertWeightMemoriesToFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
        std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
        int num_input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride,
        int num_output_channel
      );

      bool isLayerFinished(
        std::vector<std::vector<std::deque<FIFOValues>>>& inputFifos,
        std::vector<std::vector<std::deque<FIFOValues>>>& weightFifos
      );

      bool isFinishedPSumExecution(
        std::vector<PEControllerStatus>& inputControllerStatusForPEs,
        std::vector<PEControllerStatus>& weightControllerStatusForPEs
      );

      void decodeValuesToBits(
        std::vector<std::vector<std::deque<FIFOValues>>> &valueFifos,
        std::vector<DecodedRegister>& decodedRepresentations
      );

      void createInputForPEsBasedOnControllerStatus(
        std::vector<DecodedRegister>& decodedRegisters,
        std::vector<PEControllerStatus>& controllerStatusForPEs,
        std::vector<PEInput>& representationsForPEs,
        int num_Fifo
      );

      void updatePEStatus(
        std::vector<PEControllerStatus> &inputControllerStatusForPEs,
        std::vector<PEControllerStatus> &weightControllerStatusForPEs,
        std::vector<std::vector<std::deque<FIFOValues>>> &inputValuesFifos,
        std::vector<std::vector<std::deque<FIFOValues>>> &weightValuesFifos,
        std::vector<DecodedRegister> &decodedInputs,
        std::vector<DecodedRegister> &decodedWeights
      );

      void updatePEStatusWhenPsumFinish(
        std::vector<PEControllerStatus> & inputControllerStatusForPEs,
        std::vector<PEControllerStatus> & weightControllerStatusForPEs,
        std::vector<std::vector<std::deque<FIFOValues>>> & inputValuesFifos,
        std::vector<std::vector<std::deque<FIFOValues>>> & weightValuesFifos
      );

      void writeOutput(
        std::vector<std::vector<int>>& outputOfPEs,
        std::vector<std::vector<std::vector<int>>>& outputMemory,
        int outputStatus,
        int output_height,
        int output_width,
        int num_output_channel
      );

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
