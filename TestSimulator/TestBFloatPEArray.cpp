#include "gmock/gmock.h"
#include <iostream>
#include <gtest/internal/gtest-port.h>
#include <set>
#include <cstdlib>
#include "BFloatPE.h"
#include "BFloatPEArray.h"
#include "Utils.h"
#include "TestUtils.h"

using namespace testing;

namespace simulator::tests
{
  void BFloatExecOneLayer(
    int num_input_channel,
    int num_input_height,
    int num_input_width,
    int num_kernel_height,
    int num_kernel_width,
    int stride,
    int num_output_channel,
    std::set<int>& availableValueSet
  )
  {
    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    auto inputMemories = v<v<v<v<v<int>>>>>(num_PE_width, v<v<v<v<int>>>>(num_input_height, v<v<v<int>>>(num_input_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel)))));
    auto inputExpMemories = v<v<v<v<v<int>>>>>(num_PE_width, v<v<v<v<int>>>>(num_input_height, v<v<v<int>>>(num_input_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel)))));
    auto weightMemories = v<v<v<v<v<v<int>>>>>>(num_PE_height, v<v<v<v<v<int>>>>>(num_output_channel, v<v<v<v<int>>>>(num_kernel_height, v<v<v<int>>>(num_kernel_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel))))));
    auto weightExpMemories = v<v<v<v<v<v<int>>>>>>(num_PE_height, v<v<v<v<v<int>>>>>(num_output_channel, v<v<v<v<int>>>>(num_kernel_height, v<v<v<int>>>(num_kernel_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel))))));
    auto inputValues = v<v<v<int>>>(num_input_channel, v<v<int>>(num_input_height, v<int>(num_input_width)));
    auto weightValues = v<v<v<v<int>>>>(num_output_channel, v<v<v<int>>>(num_input_channel, v<v<int>>(num_kernel_height, v<int>(num_kernel_width))));
    auto outputValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));

    makeRandomInput(inputMemories, inputValues, num_input_channel, num_input_height, num_input_width, availableValueSet, stride, num_kernel_height, num_kernel_width);
    makeRandomInput(inputExpMemories, inputValues, num_input_channel, num_input_height, num_input_width, availableValueSet, stride, num_kernel_height, num_kernel_width);
    makeRandomWeight(weightMemories, weightValues, num_kernel_height, num_kernel_width, num_input_channel, num_output_channel, availableValueSet);
    makeRandomWeight(weightExpMemories, weightValues, num_kernel_height, num_kernel_width, num_input_channel, num_output_channel, availableValueSet);
    // computeConv(inputValues, weightValues, outputValues, stride);

    simulator::BFloatPEArray peArray = simulator::BFloatPEArray(
      inputMemories,
      inputExpMemories,
      weightMemories,
      weightExpMemories,
      num_input_channel,
      num_input_height,
      num_input_width,
      num_kernel_height,
      num_kernel_width,
      stride,
      num_output_channel
    );

    while (peArray.busy)
    {
      peArray.execute_one_step();
    }

    // for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
    //   for (int output_height = 0; output_height < num_output_height; output_height++){
    //     for (int output_width = 0; output_width < num_output_width; output_width++){
    //       // std::cout << outputValues[output_channel][output_height][output_width] << std::endl;
    //       // std::cout << "check real output" << std::endl;
    //       ASSERT_THAT(peArray.outputMemory[output_channel][output_height][output_width], Eq(0)) << ("output_channel, output_height, output_width = " + std::to_string(output_channel) + " " + std::to_string(output_height) + " " + std::to_string(output_width));
    //       ASSERT_THAT(peArray.outputExpMemory[output_channel][output_height][output_width], Eq(0)) << ("output_channel, output_height, output_width = " + std::to_string(output_channel) + " " + std::to_string(output_height) + " " + std::to_string(output_width));
    //     }
    //   }
    // }

    // for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
    //   for (int output_height = 0; output_height < num_output_height; output_height++){
    //     for (int output_width = 0; output_width < num_output_width; output_width++){
    //       // std::cout << outputValues[output_channel][output_height][output_width] << std::endl;
    //       // std::cout << "check real output" << std::endl;
    //       ASSERT_THAT(peArray.outputMemory[output_channel][output_height][output_width], Eq(outputValues[output_channel][output_height][output_width])) << ("output_channel, output_height, output_width = " + std::to_string(output_channel) + " " + std::to_string(output_height) + " " + std::to_string(output_width));
    //     }
    //   }
    // }
  }

  #pragma region PEArrayTests with various shape
  TEST(BFloatPEArrayTests, ExecuteMockConvManyInputOutput32){
    int num_input_channel=32;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=32;
    std::set<int> availableValueSet = {2};
    BFloatExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(BFloatPEArrayTests, ExecuteMockConvInputChannel10){
    int num_input_channel=10;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=16;
    std::set<int> availableValueSet = {2};
    BFloatExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(BFloatPEArrayTests, ExecuteMockConvDifferentSize){
    int num_input_channel=16;
    int num_input_height=4;
    int num_input_width=4;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=16;
    std::set<int> availableValueSet = {2};
    BFloatExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(BFloatPEArrayTests, ExecuteMockConvDifferentSize2){
    int num_input_channel=16;
    int num_input_height=4;
    int num_input_width=4;
    int num_kernel_height=2;
    int num_kernel_width=2;
    int stride=1;
    int num_output_channel=8;
    std::set<int> availableValueSet = {2};
    BFloatExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(BFloatPEArrayTests, ExecuteMockConvOutputChannel10){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=2;
    std::set<int> availableValueSet = {2};
    BFloatExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(BFloatPEArrayTests, ExecuteMockConvVarious){
    int num_input_channel=4096;
    int num_input_height=1;
    int num_input_width=1;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=4096;
    std::set<int> availableValueSet = {2};
    BFloatExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };
  #pragma endregion

  #pragma region PEArrayTests with pre-defined shape

  TEST(BFloatPEArrayTests, ExecuteMockConvWithConstantInput){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=8;
    std::set<int> availableValueSet = {2};
    BFloatExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(BFloatPEArrayTests, ExecuteMockConvWithZeroInput){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=8;
    std::set<int> availableValueSet = {0};
    BFloatExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  #pragma endregion

}
