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
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> inputMemories;
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>> weightMemories;
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

  TEST(PEArrayTests, ExecuteMockConv){
    int num_input_channel=32;
    int num_input_height=8;
    int num_input_width=8;
    int num_kernel_height=2;
    int num_kernel_width=2;
    int stride=1;
    int num_output_channel=16;
    std::set<int> availableValueSet = {1, 2, 3};

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    auto inputMemories = v<v<v<v<v<std::int8_t>>>>>(num_PE_width, v<v<v<v<std::int8_t>>>>(num_input_height, v<v<v<std::int8_t>>>(num_input_width, v<v<std::int8_t>>(num_input_channel, v<std::int8_t>(num_PE_parallel)))));
    auto weightMemories = v<v<v<v<v<v<std::int8_t>>>>>>(num_PE_height, v<v<v<v<v<std::int8_t>>>>>(num_output_channel, v<v<v<v<std::int8_t>>>>(num_kernel_height, v<v<v<std::int8_t>>>(num_kernel_width, v<v<std::int8_t>>(num_input_channel, v<std::int8_t>(num_PE_parallel))))));
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

    ASSERT_THAT(0, Eq(0));
  };
}
