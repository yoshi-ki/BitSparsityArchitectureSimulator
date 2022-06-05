#include "gmock/gmock.h"
#include <iostream>
#include <gtest/internal/gtest-port.h>
#include <set>
#include <cstdlib>
#include "PE.h"
#include "PEArray.h"
#include "Utils.h"
#include "TestUtils.h"

using namespace testing;

namespace simulator::tests
{
	TEST(PEArrayTests, InitializePEArray) {
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> inputMemories;
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> weightMemories;
    int num_input_channel;
    int num_input_height;
    int num_input_width;
    int num_kernel_height;
    int num_kernel_width;
    int stride;
    int num_output_channel;

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
    ASSERT_THAT(0, Eq(0));
	}

  // TEST(PEArrayTests, convertInputMemoriesToFifos){
  //   convertInputMemoriesToFifos(inputMemories, inputValuesFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);
  // }

  TEST(PEArrayTests, decodeValuesToBits){
    auto valueFifos = v<v<std::deque<FIFOValues>>>(num_PE_width, v<std::deque<FIFOValues>>(num_PE_parallel));
    auto decodedInputs = v<DecodedRegister>(num_PE_width, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_PE_parallel)), v<v<bool>>(num_PE_parallel, v<bool>(8)), v<v<bool>>(num_PE_parallel, v<bool>(8))});
    // input 5
    for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        for (int i = 0; i < 4; i++)
        {
          valueFifos[memoryIndex][bitIndex].push_back(FIFOValues{5,false});
        }
      }
    }
    auto peArray = PEArray();
    peArray.decodeValuesToBits(valueFifos, decodedInputs);
    // std::cout << "actual" << bitInputs[0][0][0] << std::endl;
    for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        ASSERT_THAT(decodedInputs[memoryIndex].bitInputValues[bitIndex][0], Eq(2)); // for 4
        ASSERT_THAT(decodedInputs[memoryIndex].isNegatives[bitIndex][0], false); // for 4
        ASSERT_THAT(decodedInputs[memoryIndex].isValids[bitIndex][0], true); // for 4
        ASSERT_THAT(decodedInputs[memoryIndex].bitInputValues[bitIndex][1], Eq(0)); // for 1
        ASSERT_THAT(decodedInputs[memoryIndex].isNegatives[bitIndex][1], false); // for 1
        ASSERT_THAT(decodedInputs[memoryIndex].isValids[bitIndex][1], true); // for 1
        ASSERT_THAT(decodedInputs[memoryIndex].isValids[bitIndex][2], false); // for nothing (for just now and needs to be updated)
      }
    }

    // input -3
    valueFifos = v<v<std::deque<FIFOValues>>>(num_PE_width, v<std::deque<FIFOValues>>(num_PE_parallel));
    decodedInputs = v<DecodedRegister>(num_PE_width, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_PE_parallel)), v<v<bool>>(num_PE_parallel, v<bool>(8)), v<v<bool>>(num_PE_parallel, v<bool>(8))});
    for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        for (int i = 0; i < 4; i++)
        {
          valueFifos[memoryIndex][bitIndex].push_back(FIFOValues{-3,false});
        }
      }
    }
    peArray.decodeValuesToBits(valueFifos, decodedInputs);
    // std::cout << "actual" << bitInputs[0][0][0] << std::endl;
    for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        ASSERT_THAT(decodedInputs[memoryIndex].bitInputValues[bitIndex][0], Eq(1)); // for 2
        ASSERT_THAT(decodedInputs[memoryIndex].isNegatives[bitIndex][0], true); // for 2
        ASSERT_THAT(decodedInputs[memoryIndex].isValids[bitIndex][0], true); // for 2
        ASSERT_THAT(decodedInputs[memoryIndex].bitInputValues[bitIndex][1], Eq(0)); // for 1
        ASSERT_THAT(decodedInputs[memoryIndex].isNegatives[bitIndex][1], true); // for 1
        ASSERT_THAT(decodedInputs[memoryIndex].isValids[bitIndex][1], true); // for 1
        ASSERT_THAT(decodedInputs[memoryIndex].isValids[bitIndex][2], false); // for nothing (for just now and needs to be updated)
      }
    }
  }

  void ExecOneLayer(
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
    auto weightMemories = v<v<v<v<v<v<int>>>>>>(num_PE_height, v<v<v<v<v<int>>>>>(num_output_channel, v<v<v<v<int>>>>(num_kernel_height, v<v<v<int>>>(num_kernel_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel))))));
    auto inputValues = v<v<v<int>>>(num_input_channel, v<v<int>>(num_input_height, v<int>(num_input_width)));
    auto weightValues = v<v<v<v<int>>>>(num_output_channel, v<v<v<int>>>(num_input_channel, v<v<int>>(num_kernel_height, v<int>(num_kernel_width))));
    auto outputValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));

    makeRandomInput(inputMemories, inputValues, num_input_channel, num_input_height, num_input_width, availableValueSet, stride, num_kernel_height, num_kernel_width);
    makeRandomWeight(weightMemories, weightValues, num_kernel_height, num_kernel_width, num_input_channel, num_output_channel, availableValueSet);
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

    while (peArray.busy)
    {
      peArray.execute_one_step();
    }

    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int output_height = 0; output_height < num_output_height; output_height++){
        for (int output_width = 0; output_width < num_output_width; output_width++){
          std::cout << outputValues[output_channel][output_height][output_width] << std::endl;
          // std::cout << "check real output" << std::endl;
          ASSERT_THAT(peArray.outputMemory[output_channel][output_height][output_width], Eq(outputValues[output_channel][output_height][output_width])) << ("output_channel, output_height, output_width = " + std::to_string(output_channel) + " " + std::to_string(output_height) + " " + std::to_string(output_width));
        }
      }
    }
  }

  #pragma region PEArrayTests with various shape
  TEST(PEArrayTests, ExecuteMockConvManyInputOutput32){
    int num_input_channel=32;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=32;
    std::set<int> availableValueSet = {2};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(PEArrayTests, ExecuteMockConvInputChannel10){
    int num_input_channel=10;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=16;
    std::set<int> availableValueSet = {2};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(PEArrayTests, ExecuteMockConvDifferentSize){
    int num_input_channel=16;
    int num_input_height=4;
    int num_input_width=4;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=16;
    std::set<int> availableValueSet = {2};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(PEArrayTests, ExecuteMockConvDifferentSize2){
    int num_input_channel=16;
    int num_input_height=4;
    int num_input_width=4;
    int num_kernel_height=2;
    int num_kernel_width=2;
    int stride=1;
    int num_output_channel=8;
    std::set<int> availableValueSet = {2};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(PEArrayTests, ExecuteMockConvOutputChannel10){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=2;
    std::set<int> availableValueSet = {2};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };
  #pragma endregion

  #pragma region PEArrayTests with pre-defined shape

  TEST(PEArrayTests, ExecuteMockConvWithConstantInput){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=8;
    std::set<int> availableValueSet = {2};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(PEArrayTests, ExecuteMockConvWithRandomInputIncludingMinus){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=8;
    std::set<int> availableValueSet = {-1,2,-2};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(PEArrayTests, ExecuteMockConvWithConstantInput3){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=8;
    std::set<int> availableValueSet = {3};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  TEST(PEArrayTests, ExecuteMockConvWithRandomInputDifferentBitSize){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=1;
    int num_kernel_width=1;
    int stride=1;
    int num_output_channel=8;
    std::set<int> availableValueSet = {0,1,2,3,4,5};
    ExecOneLayer(num_input_channel, num_input_height, num_input_width, num_kernel_height, num_kernel_width, stride, num_output_channel, availableValueSet);
  };

  #pragma endregion

}
