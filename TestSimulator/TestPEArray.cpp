#include "gmock/gmock.h"
#include <iostream>
#include <gtest/internal/gtest-port.h>
#include <set>
#include "PE.h"
#include "PEArray.h"

using namespace testing;

namespace simulator::tests
{
	TEST(PEArrayTests, InitializePEArray) {
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> inputMemories;
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> weightMemories;
    int num_input_channel;
    int input_height;
    int input_width;
    int kernel_height;
    int kernel_width;
    int stride;
    int num_output_channel;

    simulator::PEArray peArray = simulator::PEArray(
      inputMemories,
      weightMemories,
      num_input_channel,
      input_height,
      input_width,
      kernel_height,
      kernel_width,
      stride,
      num_output_channel
    );
    ASSERT_THAT(0, Eq(0));
	}

  TEST(PEArrayTests, ExecuteMockConv){
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> inputMemories;
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> weightMemories;
    int num_input_channel=32;
    int input_height=8;
    int input_width=8;
    int kernel_height=2;
    int kernel_width=2;
    int stride=1;
    int num_output_channel=16;
    std::set<int> availableValueSet = {1, 2, 3};
    // makeInput(inputMemories, num_input_channel, input_height, input_width, availableValueSet);
    // makeWeight(weightMemories, kernel_height, kernel_width, num_input_channel_group, num_output_channel)

    simulator::PEArray peArray = simulator::PEArray(
      inputMemories,
      weightMemories,
      num_input_channel,
      input_height,
      input_width,
      kernel_height,
      kernel_width,
      stride,
      num_output_channel
    );
  }

  void makeInput(){};
}
