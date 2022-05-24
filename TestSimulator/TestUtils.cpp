#include "PE.h"
#include "PEArray.h"
#include "Utils.h"
#include <set>
#include <cstdlib>


namespace simulator::tests
{
  void makeRandomInput(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
    std::vector<std::vector<std::vector<int>>>& inputValues,
    int num_input_channel,
    int num_input_height,
    int num_input_width,
    std::set<int>& availableValueSet
  )
  {
    // make random input
    std::srand(num_input_channel + num_input_height + num_input_width);
    int setSize = availableValueSet.size();
    for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
      for (int input_height = 0; input_height < num_input_height; input_height++){
        for (int input_width = 0; input_width < num_input_width; input_width++){
          inputValues[input_channel][input_height][input_width] = rand() % setSize;
        }
      }
    }

    // transform to the specific memory format
    convertInputToInputMemoryFormat(inputValues, inputMemories);

    return;
  };

  void makeRandomWeight(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>> &weightMemories,
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
            weightValues[output_channel][input_channel][kernel_height][kernel_width] = rand() % setSize;
          }
        }
      }
    }

    // transform to the specific memory format
    convertWeightToWeightMemoryFormat(weightValues, weightMemories);
  };

  void computeConv(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<int>>> &outputValues,
    int stride
  )
  {
    int num_input_channel = inputValues.size();
    int num_input_height = inputValues[0].size();
    int num_input_width = inputValues[0][0].size();
    int num_output_channel = weightValues.size();
    int num_kernel_height = weightValues[0][0].size();
    int num_kernel_width = weightValues[0][0][0].size();

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int output_height = 0; output_height < num_output_height; output_height++){
        for (int output_width = 0; output_width < num_output_width; output_width++){
          // when we want to write output_channel, output_height, output_width
          int outputValue = 0;
          for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
            for (int kernel_height = 0; kernel_height < num_kernel_height; kernel_height++){
              for (int kernel_width = 0; kernel_width < num_kernel_width; kernel_width++){
                outputValue += (inputValues[input_channel][output_height*stride + kernel_height][output_width*stride + kernel_width] *
                    weightValues[output_channel][input_channel][kernel_height][kernel_width]);
              }
            }
          }

          outputValues[output_channel][output_height][output_width] = outputValue;
        }
      }
    }
  };
}