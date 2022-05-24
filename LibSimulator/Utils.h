#include "PE.h"
#include "PEArray.h"

namespace simulator
{
  void convertInputToInputMemoryFormat(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> &inputMemories
  );

  void convertWeightToWeightMemoryFormat(
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>> &weightMemories
  );

}
