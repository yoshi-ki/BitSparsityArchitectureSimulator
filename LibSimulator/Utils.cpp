#include "PE.h"
#include "PEArray.h"
#include <iostream>
#include <stdexcept> // std::runtime_error
#include "math.h"

namespace simulator
{

  void convertInputToInputMemoryFormat(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> &inputMemories,
    int stride,
    int num_kernel_height,
    int num_kernel_width
  )
  {
    int num_input_channel = inputValues.size();
    int num_input_height = inputValues[0].size();
    int num_input_width = inputValues[0][0].size();
    // for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
    //   for (int input_height = 0; input_height < num_input_height; input_height++){
    //     for (int input_width = 0; input_width < num_input_width; input_width++){
    //       bool heightFirstGroup = input_height < ((num_input_height / 2) + (num_input_height % 2));
    //       bool widthFirstGroup = input_width < ((num_input_width / 2) + (num_input_width % 2));

    //       int memoryIndex = 2 * ((heightFirstGroup) ? 0 : 1) + ((widthFirstGroup) ? 0 : 1);
    //       int height = (heightFirstGroup) ? input_height : input_height - ((num_input_height / 2) + (num_input_height % 2));
    //       int width = (widthFirstGroup) ? input_width : input_width - ((num_input_width / 2) + (num_input_width % 2));
    //       int channelGroup = input_channel / num_PE_parallel;
    //       inputMemories[memoryIndex][height][width][channelGroup][input_channel % num_PE_parallel] =
    //           inputValues[input_channel][input_height][input_width];
    //     }
    //   }
    // }

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;
    std::cout << "output height, width: " << num_output_height << " " << num_output_width << std::endl;
    int firstGroupOutputHeight = num_output_height / 2 + num_output_height % 2;
    int firstGroupOutputWidth = num_output_width / 2 + num_output_width % 2;
    std::cout << "firstGroupOutput Height, Width: " << firstGroupOutputHeight << " " << firstGroupOutputWidth << std::endl;

    for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
      bool isheightFirstGroup = (memoryIndex / 2 == 0);
      bool iswidthFirstGroup = (memoryIndex % 2 == 0);

      int input_height_start = isheightFirstGroup ? 0 : firstGroupOutputHeight * stride;
      int input_height_end = isheightFirstGroup ? firstGroupOutputHeight * stride + num_kernel_height - 1: num_input_height;
      int input_width_start = iswidthFirstGroup ? 0 : firstGroupOutputWidth * stride;
      int input_width_end = iswidthFirstGroup ? firstGroupOutputWidth * stride + num_kernel_width - 1 : num_input_width;
      std::cout << input_height_start << input_height_end << std::endl;
      std::cout << input_width_start << input_width_end << std::endl;

      for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
        for (int input_height = input_height_start; input_height < input_height_end; input_height++){
          for (int input_width = input_width_start; input_width < input_width_end; input_width++){

            int height = input_height - input_height_start;
            int width = input_width - input_width_start;
            int channelGroup = input_channel / num_PE_parallel;
            inputMemories[memoryIndex][height][width][channelGroup][input_channel % num_PE_parallel] =
                inputValues[input_channel][input_height][input_width];
          }
        }
      }
    }

    // int num_input_channel_group = num_input_channel / num_PE_parallel + fmin(num_input_channel % num_PE_parallel, 1);
    // for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
    //   int partialInputHeight = (memoryIndex == 0 || memoryIndex == 3) ? num_input_height / 2 + num_input_height % 2 : num_input_height / 2;
    //   int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? num_input_width / 2 + num_input_width % 2 : num_input_width / 2;
    //   for (int windowStartHeight = 0; windowStartHeight < partialInputHeight + num_kernel_height - 1; windowStartHeight = windowStartHeight + stride){
    //     for (int windowStartWidth = 0; windowStartWidth < partialInputWidth + num_kernel_width - 1; windowStartWidth = windowStartWidth + stride){
    //       for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
    //         // input
    //         for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
    //           int height = (memoryIndex / 2 == 0) ? windowStartHeight : num_input_height / 2 + num_input_height % 2 + windowStartHeight;
    //           int width = (memoryIndex % 2 == 0) ? windowStartWidth: num_input_width / 2 + num_input_width % 2 + windowStartWidth;
    //           if (height < num_input_height && width < num_input_width && channelGroup * num_PE_parallel + input_channel < num_input_channel){
    //             inputMemories[memoryIndex][windowStartHeight][windowStartWidth][channelGroup][input_channel] =
    //             inputValues[channelGroup * num_PE_parallel + input_channel][height][width];
    //           }
    //         }
    //       }
    //     }
    //   }
    // }
  };

  void convertWeightToWeightMemoryFormat(
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> &weightMemories
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
            weightMemories[memoryIndex][output_channel / num_PE_height][kernel_height][kernel_width][channelGroup][input_channel % num_PE_parallel] =
                weightValues[output_channel][input_channel][kernel_height][kernel_width];
          }
        }
      }
    }
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
                outputValue +=
                  (inputValues[input_channel][output_height*stride + kernel_height][output_width*stride + kernel_width] *
                    weightValues[output_channel][input_channel][kernel_height][kernel_width]);
                // std::cout << outputValue << std::endl;
              }
            }
          }

          outputValues[output_channel][output_height][output_width] = outputValue;
        }
      }
    }
  };
}
