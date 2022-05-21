#pragma once
#include <vector>
#include <queue>

namespace simulator
{
  class PEArray
  {
    public:
      /* initializer */
      // what initializer does is to set the input and weight values, and layer config
      PEArray(std::vector<std::vector<std::int8_t>>& inputMemories, std::vector<std::vector<std::int8_t>>& weightMemories);

      /* method */
      bool execute_one_step();

      /* member */

      // input and output memory
      // std::vector<std::vector<std::int8_t>> inputMemories;
      // std::vector<std::vector<std::int8_t>> weightMemories;
      std::vector<std::vector<std::int8_t>> outputMemories;

      // // bit fifos
      // std::vector<std::queue<std::uint8_t>> bitInputFifos;
      // std::vector<std::queue<std::uint8_t>> bitWeightFifos;

      // busy or not, by this information, memory decides whether it sends data or not
      // bool busy;

    private:
      void convertMemoriesToBitFifos(
        std::vector<std::vector<std::int8_t>> inputMemories,
        std::vector<std::vector<std::int8_t>> weightMemories,
        std::vector<std::queue<std::int8_t>> inputValuesFifos,
        std::vector<std::queue<std::int8_t>> weightValuesFifos
      );

      void convertInputMemoriesToFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> inputMemories,
        std::vector<std::vector<std::queue<std::int8_t>>> inputValuesFifos,
        int input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride
      );

      void convertWeightMemoriesToFifos(
        std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> weightMemories,
        std::vector<std::vector<std::queue<std::int8_t>>> weightValuesFifos,
        int input_channel_group,
        int input_height,
        int input_width,
        int kernel_height,
        int kernel_width,
        int stride,
        int num_output_channel
      );

      void convertValuesFifoToBitFifo(
        std::queue<std::int8_t> valueFifo,
        std::queue<std::uint8_t> bitFifo
      );
  };
}
