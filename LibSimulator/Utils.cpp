#include "PE.h"
#include "PEArray.h"
#include <iostream>
#include <stdexcept> // std::runtime_error

namespace simulator
{
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
          int memoryIndex = ((input_width < (num_input_width/2 + num_input_width%2)) ? 1 : 0) +
              2 * ((input_height < (num_input_height/2 + num_input_width%2)) ? 1 : 0 );
          int width = (memoryIndex % 2 == 0) ? input_width : input_width + num_input_width / 2 + 1;
          int height = 0;
          int channelGroup = input_channel / num_PE_parallel;
          inputMemories[memoryIndex][height][width][channelGroup][input_channel % channelGroup] = inputValues[input_channel][input_height][input_width];
        }
      }
    }
  };

  void convertWeightToWeightMemoryFormat(
    std::vector<std::vector<std::vector<int>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> &weightMemories
  )
  {

  };
}
