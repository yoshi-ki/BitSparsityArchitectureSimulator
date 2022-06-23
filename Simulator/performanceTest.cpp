#include <cstdlib>
#include <iostream>

#include "PE.h"
#include "PEArray.h"
#include "Utils.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

namespace simulator::performanceTest{

  int makeNBitIntByBitSparsity(
    int n, // n bit values
    int bitSparsity
  )
  {
    int val = 0;
    for (int i = 0; i < n-1; i++){
      bool isOne = rand() % 100 + 1 >= bitSparsity;
      val = isOne ? (val + (1 << i)) : val;
    }

    // randomly prepare minus and plus
    bool isOne = rand() % 100 + 1 >= 50;
    val = isOne ? val - (1 << (n - 1)) : val;
    // if (bitSparsity == 0){
    //   std::cout << bitSparsity << " " << val << std::endl;
    // }
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
    int inputBitSparsity, // represents as percent of zeros
    int weightBitSparsity // represents as percent of zeros
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

    return cycles;
  }
}



int main(int argc, char** argv)
{
  // test
  //int num_layer = 2;
  //auto num_input_channels = std::vector<int> {  3,   2};
  //auto num_input_heights  = std::vector<int> {  4,   2};
  //auto num_input_widths   = std::vector<int> {  4,   2};
  //auto num_kernel_heights = std::vector<int> {  3,   1};
  //auto num_kernel_widths  = std::vector<int> {  3,   1};
  //auto strides            = std::vector<int> {  1,   1};
  //auto num_output_channels = std::vector<int> {64, 128};

  // VGG11 * ImageNet
  int num_layer = 11;
  auto num_input_channels = std::vector<int> {  3,  64, 128, 256, 256, 512, 512, 512, 25088, 4096, 4096};
  auto num_input_heights  = std::vector<int> { 32,  16,   8,   8,   4,   4,   2,   2,     1,    1,    1};
  auto num_input_widths   = std::vector<int> { 32, 112,   8,   8,   4,   4,   2,   2,     1,    1,    1};
  auto num_kernel_heights = std::vector<int> {  3,   3,   3,   3,   3,   3,   3,   3,     1,    1,    1};
  auto num_kernel_widths  = std::vector<int> {  3,   3,   3,   3,   3,   3,   3,   3,     1,    1,    1};
  auto strides            = std::vector<int> {  1,   1,   1,   1,   1,   1,   1,   1,     1,    1,    1};
  auto num_output_channels = std::vector<int> {64, 128, 256, 256, 512, 512, 512, 512,  4096, 4096, 1000};

  auto sumCycles = std::vector<int>(15);
  int itr = 0;
  for (int sparsity = 85; sparsity <= 99; sparsity = sparsity + 1)
  {
    int sumCycle = 0;
    for (int layer = 0; layer < num_layer; layer++){
      int cycle =
          simulator::performanceTest::ExecOneLayerOutputCount(
              num_input_channels[layer],
              num_input_heights[layer],
              num_input_widths[layer],
              num_kernel_heights[layer],
              num_kernel_widths[layer],
              strides[layer],
              num_output_channels[layer],
              sparsity,
              sparsity);
      sumCycle = sumCycle + cycle;
    }
    sumCycles[itr] = sumCycle;
    std::cout << "sparsity: " << sparsity << " cycle: " << sumCycle << std::endl;
  }
  //plt::plot(sumCycles);
  //plt::save("vgg11.pdf");
  // plt::show();

  return EXIT_SUCCESS;
}

