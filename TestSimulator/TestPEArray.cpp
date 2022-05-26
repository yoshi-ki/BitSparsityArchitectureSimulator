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

  TEST(PEArrayTests, decodeValuesToBits){
    auto valueFifos = v<v<std::deque<FIFOValues>>>(num_PE_width, v<std::deque<FIFOValues>>(num_PE_parallel));
    auto bitInputs = v<v<v<unsigned int>>>(num_PE_width, v<v<unsigned int>>(num_PE_parallel, v<unsigned int>(num_bit_size)));
    for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        for (int i = 0; i < 4; i++)
        {
          valueFifos[memoryIndex][bitIndex].push_back(FIFOValues{5,false});
        }
      }
    }
    auto peArray = PEArray();
    peArray.decodeValuesToBits(valueFifos, bitInputs);
    for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        ASSERT_THAT(bitInputs[memoryIndex][bitIndex][0], Eq(0));
        ASSERT_THAT(bitInputs[memoryIndex][bitIndex][1], Eq(2));
        ASSERT_THAT(bitInputs[memoryIndex][bitIndex][2], Eq(0));
      }
    }
  }

  TEST(PEArrayTests, ExecuteMockConv){
    int num_input_channel=16;
    int num_input_height=2;
    int num_input_width=2;
    int num_kernel_height=2;
    int num_kernel_width=2;
    int stride=1;
    int num_output_channel=16;
    std::set<int> availableValueSet = {1, 2, 3};

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    auto inputMemories = v<v<v<v<v<int>>>>>(num_PE_width, v<v<v<v<int>>>>(num_input_height, v<v<v<int>>>(num_input_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel)))));
    auto weightMemories = v<v<v<v<v<v<int>>>>>>(num_PE_height, v<v<v<v<v<int>>>>>(num_output_channel, v<v<v<v<int>>>>(num_kernel_height, v<v<v<int>>>(num_kernel_width, v<v<int>>(num_input_channel, v<int>(num_PE_parallel))))));
    auto inputValues = v<v<v<int>>>(num_input_channel, v<v<int>>(num_input_height, v<int>(num_input_width)));
    auto weightValues = v<v<v<v<int>>>>(num_output_channel, v<v<v<int>>>(num_input_channel, v<v<int>>(num_kernel_height, v<int>(num_kernel_width))));
    auto outputValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));

    makeRandomInput(inputMemories, inputValues, num_input_channel, num_input_height, num_input_width, availableValueSet);
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

    while(peArray.busy){
      peArray.execute_one_step();
    }

    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int output_height = 0; output_height < num_output_height; output_height++){
        for (int output_width = 0; output_width < num_output_width; num_output_width++){
          ASSERT_THAT(peArray.outputMemory[output_channel][output_height][output_width], Eq(outputValues[output_channel][output_height][output_width])) << ("output_channel, output_height, output_width = " + std::to_string(output_channel) + " " + std::to_string(output_height) + " " + std::to_string(output_width));
        }
      }
    }

      ASSERT_THAT(0, Eq(0));
  };
}
