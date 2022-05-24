#include "PE.h"
#include "PEArray.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

namespace simulator
{
  // int num_PE_parallel = 16;
  // int num_PE_height = 8;

  void convertInputToInputMemoryFormat(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> &inputMemories
  )
  {
    int num_input_channel = inputValues.size();
    int num_input_height = inputValues[0].size();
    int num_input_width = inputValues[0][0].size();
    for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
      for (int input_height = 0; input_height < num_input_height; input_height++){
        for (int input_width = 0; input_width < num_input_width; input_width++){
          bool heightFirstGroup = input_height < (num_input_height / 2) + (num_input_height % 2);
          bool widthFirstGroup = input_width < (num_input_width / 2) + (num_input_width % 2);

          int memoryIndex = 2 * ((heightFirstGroup) ? 0 : 1) + ((widthFirstGroup) ? 0 : 1);
          int height = (heightFirstGroup) ? input_height : input_height - ((num_input_height / 2) + (num_input_height % 2));
          int width = (widthFirstGroup) ? input_width : input_width - ((num_input_height / 2) + (num_input_height % 2));
          int channelGroup = input_channel / num_PE_parallel;
          inputMemories[memoryIndex][height][width][channelGroup][input_channel % num_PE_parallel] =
              inputValues[input_channel][input_height][input_width];
        }
      }
    }
  };

  void convertWeightToWeightMemoryFormat(
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>> &weightMemories
  )
  {
    int num_output_channel = weightValues.size();
    int num_input_channel = weightValues[0].size();
    int num_kernel_height = weightValues[0][0].size();
    int num_kernel_width = weightValues[0][0][0].size();
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
        for (int kernel_height = 0; kernel_height < num_kernel_height; kernel_height++){
          for (int kernel_width = 0; kernel_width < num_kernel_width; kernel_width++){
            int memoryIndex = output_channel % num_PE_height;
            int channelGroup = input_channel / num_PE_parallel;
            weightMemories[memoryIndex][output_channel / num_PE_height][kernel_height][kernel_width][channelGroup][input_channel % channelGroup] =
                weightValues[output_channel][input_channel][kernel_height][kernel_width];
          }
        }
      }
    }
  };
}
