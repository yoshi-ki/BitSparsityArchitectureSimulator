#include "PE.h"
#include "PEArray.h"

namespace simulator
{
  void convertInputToInputMemoryFormat(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> &inputMemories,
    int stride,
    int num_kernel_height,
    int num_kernel_width
  );

  void convertWeightToWeightMemoryFormat(
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> &weightMemories
  );

  void computeConv(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<int>>> &outputValues,
    int stride
  );

}
