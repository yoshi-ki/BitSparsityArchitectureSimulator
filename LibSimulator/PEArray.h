#pragma once
#include <vector>
#include <queue>

namespace simulator
{
  struct PEControllerStatus{
    std::vector<bool> isWaiting;
    std::vector<int> nextProcessIndex;
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
      std::vector<std::vector<std::int8_t>> outputMemories;

      // value fifos
      std::vector<std::vector<std::queue<std::int8_t>>> inputValuesFifos;
      std::vector<std::vector<std::queue<std::int8_t>>> weightValuesFifos;

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
      void convertMemoriesToBitFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& weightMemories,
        std::vector<std::vector<std::queue<std::uint8_t>>>& inputValuesFifos,
        std::vector<std::vector<std::queue<std::uint8_t>>>& weightValuesFifos,
        int num_input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride,
        int num_output_channel
      );

      void convertInputMemoriesToFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
        std::vector<std::vector<std::queue<std::int8_t>>>& inputValuesFifos,
        int num_input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride
      );

      void convertWeightMemoriesToFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& weightMemories,
        std::vector<std::vector<std::queue<std::int8_t>>>& weightValuesFifos,
        int num_input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride,
        int num_output_channel
      );

      void convertValuesFifosToBitFifos(
        std::vector<std::vector<std::queue<std::int8_t>>>& valuesFifos,
        std::vector<std::vector<std::queue<std::uint8_t>>>& bitFifos
      );

      bool PEArray::isLayerFinished(
        std::vector<std::vector<std::queue<std::int8_t>>>& inputFifos,
        std::vector<std::vector<std::queue<std::int8_t>>>& weightFifos
      );

      void PEArray::decodeValuesToBits(
        std::vector<std::vector<std::queue<std::int8_t>>> &valueFifos,
        std::vector<std::vector<std::vector<std::uint8_t>>> bitRepresentations
      );

      void PEArray::createInputForPEsBasedOnControllerStatus(
        std::vector<std::vector<std::vector<std::uint8_t>>>& bitRepresentations,
        std::vector<PEControllerStatus>& controllerStatusForPEs,
        std::vector<std::vector<unsigned int>>& representationsForPEs
      );
  };
}
