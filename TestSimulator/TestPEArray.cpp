#include "gmock/gmock.h"
#include <iostream>
#include <gtest/internal/gtest-port.h>
#include <set>
#include <cstdlib>
#include "PE.h"
#include "PEArray.h"
#include "Utils.h"

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
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> weightMemories;
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
    makeRandomWeight(weightMemories, weightValues, num_kernel_height, num_kernel_width, num_input_channel, num_output_channel);
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
  }

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
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>> &weightMemories,
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
    for (int output_channel = 0; output_channel < num_output_channel; output_channel){
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
  }

  void computeConv(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<int>>> &outputValues,
    int stride
  )
  {
    // TODO: compute conv here
  };
}
