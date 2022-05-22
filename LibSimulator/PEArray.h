#pragma once
#include <vector>
#include <deque>

namespace simulator
{
  struct PEControllerStatus{
    std::vector<bool> isWaiting;
    std::vector<int> nextProcessIndex;
    std::vector<bool> finishedPSum;
  };

  struct FIFOValues{
    std::int8_t value;
    bool isLast;
  };

  class PEArray
  {
    public:
      /* initializer */
      // what initializer does is to set the input and weight values, and layer config
      PEArray(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& weightMemories,
        int num_input_channel_group,
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
      // std::vector<std::vector<std::int8_t>> inputMemories;
      // std::vector<std::vector<std::int8_t>> weightMemories;
      std::vector<std::vector<std::vector<int>>> outputMemory;
      int outputStatus;

      // value fifos
      std::vector<std::vector<std::deque<FIFOValues>>> inputValuesFifos;
      std::vector<std::vector<std::deque<FIFOValues>>> weightValuesFifos;

      // bit fifos
      std::vector<std::vector<std::vector<std::uint8_t>>> bitInputs;
      std::vector<std::vector<std::vector<std::uint8_t>>> bitWeights;

      // states to control PEs
      // std::vector<bool> isInputFifoWaiting;
      // std::vector<bool> isWeightFifoWaiting;
      // std::vector<int> indexInputBit;
      // std::vector<int> indexWeightBit;
      std::vector<PEControllerStatus> inputControllerStatusForPEs;
      std::vector<PEControllerStatus> weightControllerStatusForPEs;

      // busy or not, by this information, memory decides whether it sends data or not
      bool busy;

    private:
      void convertInputMemoriesToFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
        std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
        int num_input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride
      );

      void convertWeightMemoriesToFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& weightMemories,
        std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
        int num_input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride,
        int num_output_channel
      );

      bool PEArray::isLayerFinished(
        std::vector<std::vector<std::deque<FIFOValues>>>& inputFifos,
        std::vector<std::vector<std::deque<FIFOValues>>>& weightFifos
      );

      bool PEArray::isFinishedPSumExecution(
        std::vector<PEControllerStatus> inputControllerStatusForPEs,
        std::vector<PEControllerStatus> weightControllerStatusForPEs
      );

      void PEArray::decodeValuesToBits(
        std::vector<std::vector<std::deque<FIFOValues>>> &valueFifos,
        std::vector<std::vector<std::vector<std::uint8_t>>> bitRepresentations
      );

      void PEArray::createInputForPEsBasedOnControllerStatus(
        std::vector<std::vector<std::vector<std::uint8_t>>>& bitRepresentations,
        std::vector<PEControllerStatus>& controllerStatusForPEs,
        std::vector<std::vector<unsigned int>>& representationsForPEs
      );

      void PEArray::updatePEStatus(
        std::vector<PEControllerStatus> &inputControllerStatusForPEs,
        std::vector<PEControllerStatus> &weightControllerStatusForPEs,
        std::vector<std::vector<std::deque<FIFOValues>>> &inputValuesFifos,
        std::vector<std::vector<std::deque<FIFOValues>>> &weightValuesFifos,
        std::vector<std::vector<std::vector<std::uint8_t>>> &bitInputs,
        std::vector<std::vector<std::vector<std::uint8_t>>> &bitWeights
      );

      void PEArray::updatePEStatusWhenPsumFinish(
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
