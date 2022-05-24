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
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> inputMemories;
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>> weightMemories;
    int num_input_channel=32;
    int num_input_height=8;
    int num_input_width=8;
    int num_kernel_height=2;
    int num_kernel_width=2;
    int stride=1;
    int num_output_channel=16;
    std::set<int> availableValueSet = {1, 2, 3};

    std::vector<std::vector<std::vector<int>>> inputValues;
    std::vector<std::vector<std::vector<std::vector<int>>>> weightValues;
    std::vector<std::vector<std::vector<int>>> outputValues;

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
