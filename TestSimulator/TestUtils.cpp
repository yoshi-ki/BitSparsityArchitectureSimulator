#include "PE.h"
#include "PEArray.h"
#include "Utils.h"
#include <set>
#include <cstdlib>
#include <iostream>


namespace simulator::tests
{
  void makeRandomInput(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::vector<int>>>& inputValues,
    int num_input_channel,
    int num_input_height,
    int num_input_width,
    std::set<int>& availableValueSet,
    int stride,
    int num_kernel_height,
    int num_kernel_width
  )
  {
    // make random input
    std::srand(num_input_channel + num_input_height + num_input_width);
    int setSize = availableValueSet.size();
    for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
      for (int input_height = 0; input_height < num_input_height; input_height++){
        for (int input_width = 0; input_width < num_input_width; input_width++){
          // inputValues[input_channel][input_height][input_width] = rand() % setSize;
          inputValues[input_channel][input_height][input_width] = *std::next(availableValueSet.begin(), rand()%setSize);
          // inputValues[input_channel][input_height][input_width] = 2;
        }
      }
    }

    // transform to the specific memory format
    convertInputToInputMemoryFormat(inputValues, inputMemories, stride, num_kernel_height, num_kernel_width);

    return;
  };

  void makeRandomWeight(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> &weightMemories,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    int num_kernel_height,
    int num_kernel_width,
    int num_input_channel,
    int num_output_channel,
    std::set<int> & availableValueSet
  )
  {
    // make random weight
    std::srand(num_kernel_height + num_kernel_width + num_input_channel + num_output_channel);
    int setSize = availableValueSet.size();
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
        for (int kernel_height = 0; kernel_height < num_kernel_height; kernel_height++){
          for (int kernel_width = 0; kernel_width < num_kernel_width; kernel_width++){
            // weightValues[output_channel][input_channel][kernel_height][kernel_width] = rand() % setSize;
            weightValues[output_channel][input_channel][kernel_height][kernel_width] = *std::next(availableValueSet.begin(), rand()%setSize);
            // weightValues[output_channel][input_channel][kernel_height][kernel_width] = 2;
          }
        }
      }
    }

    // transform to the specific memory format
    convertWeightToWeightMemoryFormat(weightValues, weightMemories);
  };
}