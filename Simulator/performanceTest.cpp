#include <cstdlib>
#include <iostream>

#include "PE.h"
#include "PEArray.h"
#include "Utils.h"

namespace simulator::performanceTest{

  int makeNBitIntByBitSparsity(
    int n, // n bit values
    int bitSparsity
  )
  {
    int val = 0;
    for (int i = 0; i < n-1; i++){
      bool isOne = rand() % 100 + 1 >= bitSparsity;
      val = isOne ? (val + 2 ^ i) : val;
    }
    bool isOne = rand() % 100 + 1 >= bitSparsity;
    val = isOne ? val - 2 ^ (n - 1) : val;
    return val;
  }
  void makeSparseInputWithPercent(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::vector<int>>>& inputValues,
    int num_input_channel,
    int num_input_height,
    int num_input_width,
    int stride,
    int num_kernel_height,
    int num_kernel_width,
    int inputBitSparsity
  )
  {
    // make random input
    std::srand(num_input_channel + num_input_height + num_input_width);
    for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
      for (int input_height = 0; input_height < num_input_height; input_height++){
        for (int input_width = 0; input_width < num_input_width; input_width++){
          inputValues[input_channel][input_height][input_width] = makeNBitIntByBitSparsity(8, inputBitSparsity);
        }
      }
    }

    // transform to the specific memory format
    convertInputToInputMemoryFormat(inputValues, inputMemories, stride, num_kernel_height, num_kernel_width);

    return;
  };

  void makeSparseWeightWithPercent(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> &weightMemories,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    int num_kernel_height,
    int num_kernel_width,
    int num_input_channel,
    int num_output_channel,
    int weightBitSparsity
  )
  {
    // make random weight
    std::srand(num_kernel_height + num_kernel_width + num_input_channel + num_output_channel);
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
        for (int kernel_height = 0; kernel_height < num_kernel_height; kernel_height++){
          for (int kernel_width = 0; kernel_width < num_kernel_width; kernel_width++){
            weightValues[output_channel][input_channel][kernel_height][kernel_width] = makeNBitIntByBitSparsity(8, weightBitSparsity);
          }
        }
      }
    }

    // transform to the specific memory format
    convertWeightToWeightMemoryFormat(weightValues, weightMemories);
  };

  int ExecOneLayerOutputCount(
    int num_input_channel,
    int num_input_height,
    int num_input_width,
    int num_kernel_height,
    int num_kernel_width,
    int stride,
    int num_output_channel,
    float inputBitSparsity, // represents as percent of zeros
    float weightBitSparsity // represents as percent of zeros
  )
  {
    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    auto inputMemories = v<v<v<v<v<int>>>>>(num_PE_width, v<v<v<v<int>>>>(num_input_height, v<v<v<int>>>(num_input_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel)))));
    auto weightMemories = v<v<v<v<v<v<int>>>>>>(num_PE_height, v<v<v<v<v<int>>>>>(num_output_channel, v<v<v<v<int>>>>(num_kernel_height, v<v<v<int>>>(num_kernel_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel))))));
    auto inputValues = v<v<v<int>>>(num_input_channel, v<v<int>>(num_input_height, v<int>(num_input_width)));
    auto weightValues = v<v<v<v<int>>>>(num_output_channel, v<v<v<int>>>(num_input_channel, v<v<int>>(num_kernel_height, v<int>(num_kernel_width))));
    auto outputValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));

    makeSparseInputWithPercent(inputMemories, inputValues, num_input_channel, num_input_height, num_input_width, stride, num_kernel_height, num_kernel_width, inputBitSparsity);
    makeSparseWeightWithPercent(weightMemories, weightValues, num_kernel_height, num_kernel_width, num_input_channel, num_output_channel, weightBitSparsity);
    computeConv(inputValues, weightValues, outputValues, stride);

    simulator::PEArray peArray = simulator::PEArray(
      inputMemories,
      weightMemories,
      num_input_channel,
      num_input_height,
      num_input_width,
      num_kernel_height,
      num_kernel_width,
      stride,
      num_output_channel
    );

    int cycles = 0;
    while (peArray.busy)
    {
      peArray.execute_one_step();
      cycles++;
    }

    // for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
    //   for (int output_height = 0; output_height < num_output_height; output_height++){
    //     for (int output_width = 0; output_width < num_output_width; output_width++){
    //       std::cout << outputValues[output_channel][output_height][output_width] << std::endl;
    //     }
    //   }
    // }

    return cycles;
  }
}

int main(int argc, char** argv)
{
  std::cout << "hello world => " << simulator::performanceTest::ExecOneLayerOutputCount(16,4,4,2,2,1,8,30,30) << std::endl;

  return EXIT_SUCCESS;
}

