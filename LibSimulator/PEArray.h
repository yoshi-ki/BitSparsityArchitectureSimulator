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
      PEArray();

      /* method */
      int execute_one_step(std::vector<unsigned int>& bitActivations, std::vector<unsigned int>& bitWeights);

      // for test and mock
      // output: 22bit signed
      int get_psum();


      /* member */

      // input and output memory
      std::vector<std::vector<std::int8_t>> inputMemories;
      std::vector<std::vector<std::int8_t>> weightMemories;
      std::vector<std::vector<std::int8_t>> outputMemories;

      // bit fifos
      std::vector<std::queue<std::uint8_t>> bitInputFifos;
      std::vector<std::queue<std::uint8_t>> bitWeightFifos;

      // busy or not, by this information, memory decides whether it sends data or not
      bool busy;

    private:
      void convertMemoriesToBitFifos(
        std::vector<std::vector<std::int8_t>> inputMemories,
        std::vector<std::vector<std::int8_t>> weightMemories,
        std::vector<std::queue<std::uint8_t>> bitInputFifos,
        std::vector<std::queue<std::uint8_t>> bitWeightFifos
      );

      void convertValuesFifoToBitFifo(
        std::queue<std::int8_t> valueFifo,
        std::queue<std::uint8_t> bitFifo
      );
  };
}
