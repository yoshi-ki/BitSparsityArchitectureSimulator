#include <cstdlib>
#include <iostream>
#include <math.h>

#include "BFloatPE.h"
#include "BFloatPEArray.h"
#include "Utils.h"

#include <torch/torch.h>
#include <torch/script.h>

std::pair<int,int> CreateBFloatFromFloat(float v){
  union int_float{
        uint32_t i;
        float f;
    } tofloat;

  uint32_t input;
  tofloat.f = v;
  input = tofloat.i;
  uint16_t output;

  if (std::isnan(v)) {
    // If the value is a NaN, squash it to a qNaN with msb of fraction set,
    // this makes sure after truncation we don't end up with an inf.
    //
    // qNaN magic: All exponent bits set + most significant bit of fraction
    // set.
    output = 0x7fc0;
  } else {
    uint32_t lsb = (input >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    input += rounding_bias;
    output = static_cast<uint16_t>(input >> 16);
  }
  // output is bfloat, we will convert it to our format, exp and mantissa (manttisa includes sign)
  int exp = (output >> 7) & 0b11111111;
  int mantissa = (output & 0b1111111) + ((output >> 15) & 0b1) * 128;

  // std::cout << std::bitset<16>(output) << " " << std::bitset<8>(exp) << " " << std::bitset<8>(mantissa) << std::endl;

  return std::make_pair(exp, mantissa);
}


float CreateBFloatFromFloat(std::pair<int,int> p){
  // create float from bfloat
  int exp = p.first;
  int mantissa = p.second;
  int exp2 = (exp & 0b11111111) + (mantissa >> 7 & 0b1) * 256;
  uint8_t mantissa2 = (((uint8_t)mantissa) << 1);
  mantissa2 = mantissa2 >> 1;
  uint32_t target = (exp2 << 7 & 0b1111111110000000) + (uint32_t)mantissa2;
  uint32_t target2 = target << 16;
  union int_float{
        uint32_t i;
        float f;
    } tofloat;
  tofloat.i = target2;

  // std::cout << std::bitset<32>(exp) << " " << std::bitset<8>(mantissa) << std::endl;
  // std::cout << std::bitset<32>(exp2) << " " << std::bitset<8>(mantissa2) << std::endl;
  // std::cout << std::bitset<32>(target) << " " << std::bitset<32>(target2) << std::endl;

  return tofloat.f;
}

void CreateBFloatWeight(v<v<v<v<float>>>>& weightVector, v<v<v<v<int>>>>& weightExpVector, v<v<v<v<int>>>>& weightMantissaVector)
{
  for (int output_channel = 0; output_channel < weightVector.size(); output_channel++){
    for (int input_channel = 0; input_channel < weightVector[0].size(); input_channel++){
      for (int kernel_height = 0; kernel_height < weightVector[0][0].size(); kernel_height++){
        for (int kernel_width = 0; kernel_width < weightVector[0][0][0].size(); kernel_width++){
          // create bfloat value
          auto p = CreateBFloatFromFloat(weightVector[output_channel][input_channel][kernel_height][kernel_width]);
          weightExpVector[output_channel][input_channel][kernel_height][kernel_width] = p.first;
          weightMantissaVector[output_channel][input_channel][kernel_height][kernel_width] = p.second;
        }
      }
    }
  }
  return;
}

void CreateBFloatInput(v<v<v<float>>>& inputValues, v<v<v<int>>>& inputExpValues, v<v<v<int>>>& inputMantissaValues){
  for (int input_channel = 0; input_channel < inputValues.size(); input_channel++){
    for (int input_height = 0; input_height < inputValues[0].size(); input_height++){
      for (int input_width = 0; input_width < inputValues[0][0].size(); input_width++){
        // create bfloat value
        auto p = CreateBFloatFromFloat(inputValues[input_channel][input_height][input_width]);
        inputExpValues[input_channel][input_height][input_width] = p.first;
        inputMantissaValues[input_channel][input_height][input_width] = p.second;
      }
    }
  }
  return;
}

int ExecNeuralNetwork(v<v<v<int>>>& inputExpValues, v<v<v<int>>>& inputMantissaValues, v<v<v<v<v<int>>>>>& weightExpVectors, v<v<v<v<v<int>>>>>& weightMantissaVectors){
  v<v<v<int>>> layerInputExpValues;
  // initialize first layer input
  for (int input_channel = 0; input_channel < inputExpValues.size(); input_channel++){
    v<v<int>> temp1;
    for (int input_height = 0; input_height < inputExpValues[0].size(); input_height++){
      v<int> temp2;
      for (int input_width = 0; input_width < inputExpValues[0][0].size(); input_width++){
        temp2.push_back(inputExpValues[input_channel][input_height][input_width]);
      }
      temp1.push_back(temp2);
    }
    layerInputExpValues.push_back(temp1);
  }

  v<v<v<int>>> layerInputMantissaValues;
  // initialize first layer input
  for (int input_channel = 0; input_channel <inputMantissaValues.size(); input_channel++){
    v<v<int>> temp1;
    for (int input_height = 0; input_height < inputMantissaValues[0].size(); input_height++){
      v<int> temp2;
      for (int input_width = 0; input_width < inputMantissaValues[0][0].size(); input_width++){
        temp2.push_back(inputMantissaValues[input_channel][input_height][input_width]);
      }
      temp1.push_back(temp2);
    }
    layerInputMantissaValues.push_back(temp1);
  }

  for (int layerIndex = 0; layerIndex < weightExpVectors.size(); layerIndex++)
  {
    int num_output_channel = weightExpVectors[layerIndex].size();
    int num_input_channel = weightExpVectors[layerIndex][0].size();
    int num_kernel_height = weightExpVectors[layerIndex][0][0].size();
    int num_kernel_width = weightExpVectors[layerIndex][0][0][0].size();
    int num_input_height = layerInputExpValues[0].size();
    int num_input_width = layerInputExpValues[0][0].size();
    int stride = 1;

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    auto inputExpMemories = v<v<v<v<v<int>>>>>(simulator::num_PE_width, v<v<v<v<int>>>>(num_input_height, v<v<v<int>>>(num_input_width, v<v<int>>(num_input_channel, v<int>(simulator::num_PE_parallel)))));
    auto inputMantissaMemories = v<v<v<v<v<int>>>>>(simulator::num_PE_width, v<v<v<v<int>>>>(num_input_height, v<v<v<int>>>(num_input_width, v<v<int>>(num_input_channel, v<int>(simulator::num_PE_parallel)))));
    auto weightExpMemories = v<v<v<v<v<v<int>>>>>>(simulator::num_PE_height, v<v<v<v<v<int>>>>>(num_output_channel, v<v<v<v<int>>>>(num_kernel_height, v<v<v<int>>>(num_kernel_width, v<v<int>>(num_input_channel, v<int>(simulator::num_PE_parallel))))));
    auto weightMantissaMemories = v<v<v<v<v<v<int>>>>>>(simulator::num_PE_height, v<v<v<v<v<int>>>>>(num_output_channel, v<v<v<v<int>>>>(num_kernel_height, v<v<v<int>>>(num_kernel_width, v<v<int>>(num_input_channel, v<int>(simulator::num_PE_parallel))))));
    auto outputValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));
    simulator::convertInputToInputMemoryFormat(layerInputExpValues, inputExpMemories, stride, num_kernel_height, num_kernel_width);
    simulator::convertInputToInputMemoryFormat(layerInputMantissaValues, inputMantissaMemories, stride, num_kernel_height, num_kernel_width);
    simulator::convertWeightToWeightMemoryFormat(weightMantissaVectors[layerIndex], weightExpMemories);
    simulator::convertWeightToWeightMemoryFormat(weightExpVectors[layerIndex], weightMantissaMemories);

    simulator::BFloatPEArray peArray = simulator::BFloatPEArray(
      inputExpMemories,
      inputMantissaMemories,
      weightExpMemories,
      weightMantissaMemories,
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

    // output is peArray.outputMemory
    layerInputExpValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));
    layerInputMantissaValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int output_height = 0; output_height < num_output_height; output_height++){
        for (int output_width = 0; output_width < num_output_width; output_width++){
          layerInputExpValues[output_channel][output_height][output_width] = peArray.outputExpMemory[output_channel][output_height][output_width];
          layerInputMantissaValues[output_channel][output_height][output_width] = peArray.outputMemory[output_channel][output_height][output_width];
        }
      }
    }

    // relu
    // relu is applied except the last layer
    if (layerIndex != weightExpVectors.size() - 1){
      for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
        for (int output_height = 0; output_height < num_output_height; output_height++){
          for (int output_width = 0; output_width < num_output_width; output_width++){
            layerInputExpValues[output_channel][output_height][output_width] = (layerInputMantissaValues[output_channel][output_height][output_width] < 0) ? 0 : layerInputExpValues[output_channel][output_height][output_width];
            layerInputMantissaValues[output_channel][output_height][output_width] = std::max(layerInputMantissaValues[output_channel][output_height][output_width], 0);
          }
        }
      }
    }

    // x.view(-1)
    // view is applied only when layerIndex == 1, TODO: we need to update this part for generalization
    if (layerIndex == 1){
      auto newLayerInputExpValues = v<v<v<int>>>(num_output_channel*num_output_height*num_output_width, v<v<int>>(1, v<int>(1)));
      auto newLayerInputMantissaValues = v<v<v<int>>>(num_output_channel*num_output_height*num_output_width, v<v<int>>(1, v<int>(1)));
      int viewIndex = -1;
      for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
        for (int output_height = 0; output_height < num_output_height; output_height++){
          for (int output_width = 0; output_width < num_output_width; output_width++){
            viewIndex++;
            newLayerInputExpValues[viewIndex][0][0] = layerInputExpValues[output_channel][output_height][output_width];
            newLayerInputMantissaValues[viewIndex][0][0] = layerInputMantissaValues[output_channel][output_height][output_width];
          }
        }
      }
      layerInputExpValues = newLayerInputExpValues;
      layerInputMantissaValues = newLayerInputMantissaValues;
    }

    std::cout << "layer: " << layerIndex << " cycle: " << cycles << std::endl;
  }

  // std::cout << "output: " << layerInputExpValues << std::endl;
  // TODO: update this part
  std::vector<float> scores = std::vector<float>(layerInputExpValues.size());
  for (int classIndex = 0; classIndex < layerInputExpValues.size(); classIndex++)
  {
    float score = CreateBFloatFromFloat(std::make_pair(layerInputExpValues[classIndex][0][0], layerInputMantissaValues[classIndex][0][0]));
    scores[classIndex] = score;
  }
  std::cout << scores << std::endl;
  int classifiedResult = std::max_element(scores.begin(), scores.end()) - scores.begin();
  return classifiedResult;
}

int main(int argc, char** argv)
{
  #pragma region Load Weight Data
  // load weight data
  torch::jit::script::Module module;
  module = torch::jit::load(argv[1]);

  std::cout << "success" << std::endl;
  std::vector<std::string> layerSettings = {"conv", "conv1", "conv", "conv1", "fc", "fc1", "fc", "fc1"};

  auto num_input_channels = std::vector<int> {  1,  32, 36864, 512};
  auto num_input_heights  = std::vector<int> { 28,  26,   1,   1};
  auto num_input_widths   = std::vector<int> { 28,  26,   1,   1};
  auto num_kernel_heights = std::vector<int> {  3,   3,   1,   1};
  auto num_kernel_widths  = std::vector<int> {  3,   3,   1,   1};
  auto strides            = std::vector<int> {  1,   1,   1,   1};
  auto num_output_channels = std::vector<int> {32,  64, 512,  10};

  std::vector<v<v<v<v<float>>>>> weightVectors;
  std::vector<v<v<v<v<int>>>>> weightExpVectors;
  std::vector<v<v<v<v<int>>>>> weightMantissaVectors;

  int index = -1;
  int parseIndex = -1;
  for (const auto& p : module.parameters()) {
    index++;
    auto layerSetting = layerSettings[index];
    if (layerSetting == "conv")
    {
      parseIndex++;
      std::cout << p.size(0) << " " << p.size(1) << " " << p.size(2) << " " << p.size(3) << std::endl;

      auto weightValues = v<v<v<v<float>>>>(num_output_channels[parseIndex], v<v<v<float>>>(num_input_channels[parseIndex], v<v<float>>(num_kernel_heights[parseIndex], v<float>(num_kernel_widths[parseIndex]))));
      // auto weightValues = v<v<v<v<float>>>>(100, v<v<v<float>>>(100, v<v<float>>(100, v<float>(100))));
      for (int output_channel = 0; output_channel < num_output_channels[parseIndex]; output_channel++){
        for (int input_channel = 0; input_channel < num_input_channels[parseIndex]; input_channel++){
          for (int kernel_height = 0; kernel_height < num_kernel_heights[parseIndex]; kernel_height++){
            for (int kernel_width = 0; kernel_width < num_kernel_widths[parseIndex]; kernel_width++){
              // float a = p[output_channel][input_channel][kernel_height][kernel_width].item<float>();
              weightValues[output_channel][input_channel][kernel_height][kernel_width] = p[output_channel][input_channel][kernel_height][kernel_width].item<float>();
            }
          }
        }
      }
      weightVectors.push_back(weightValues);
    }
    else if (layerSetting == "fc"){
      parseIndex++;
      auto weightValues = v<v<v<v<float>>>>(num_output_channels[parseIndex], v<v<v<float>>>(num_input_channels[parseIndex], v<v<float>>(num_kernel_heights[parseIndex], v<float>(num_kernel_widths[parseIndex]))));
      for (int output_channel = 0; output_channel < num_output_channels[parseIndex]; output_channel++){
        for (int input_channel = 0; input_channel < num_input_channels[parseIndex]; input_channel++){
          // fc layer can be considered as 1x1 conv
          for (int kernel_height = 0; kernel_height < 1; kernel_height++){
            for (int kernel_width = 0; kernel_width < 1; kernel_width++){
              weightValues[output_channel][input_channel][kernel_height][kernel_width] = p[output_channel][input_channel].item<float>();
            }
          }
        }
      }
      weightVectors.push_back(weightValues);
      std::cout << p.size(0) << " " << p.size(1) << std::endl;
    }
    // std::cout << "parameter" << std::endl;
  }

  // make bfloat weight
  for (int weightVectorIndex = 0; weightVectorIndex < weightVectors.size(); weightVectorIndex++){
    auto weightExpVector = v<v<v<v<int>>>>(weightVectors[weightVectorIndex].size(), v<v<v<int>>>(weightVectors[weightVectorIndex][0].size(), v<v<int>>(weightVectors[weightVectorIndex][0][0].size(), v<int>(weightVectors[weightVectorIndex][0][0][0].size()))));
    auto weightMantissaVector = v<v<v<v<int>>>>(weightVectors[weightVectorIndex].size(), v<v<v<int>>>(weightVectors[weightVectorIndex][0].size(), v<v<int>>(weightVectors[weightVectorIndex][0][0].size(), v<int>(weightVectors[weightVectorIndex][0][0][0].size()))));
    CreateBFloatWeight(weightVectors[weightVectorIndex], weightExpVector, weightMantissaVector);
    weightExpVectors.push_back(weightExpVector);
    weightMantissaVectors.push_back(weightMantissaVector);
  }
  #pragma endregion Load Weight Data


  #pragma region Load Input Data
  // load input data and process
  auto dataset = torch::data::datasets::MNIST("./mnist")
    .map(torch::data::transforms::Normalize<>(0, 0.5))
    .map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader(std::move(dataset));

  int num_input_channel = 1;
  int num_input_height = 28;
  int num_input_width = 28;
  auto inputValues = v<v<v<float>>>(num_input_channel, v<v<float>>(num_input_height, v<float>(num_input_width)));
  auto inputExpValues = v<v<v<int>>>(num_input_channel, v<v<int>>(num_input_height, v<int>(num_input_width)));
  auto inputMantissaValues = v<v<v<int>>>(num_input_channel, v<v<int>>(num_input_height, v<int>(num_input_width)));

  int maxSample = 100;
  int correctSample = 0;
  int sample = 0;
  for (torch::data::Example<> &batch : *data_loader)
  {
    std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    for (int64_t i = 0; i < batch.data.size(0); ++i) {
      int targetResult = batch.target[i].item<int>();
      std::cout << targetResult << std::endl;
      // std::cout << batch.data << std::endl;
      for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
        for (int input_height = 0; input_height < num_input_height; input_height++){
          for (int input_width = 0; input_width < num_input_width; input_width++){
            // std::cout << batch.data.size(0) << " " << batch.data.size(1) << " " << batch.data.size(2) << " " << batch.data.size(3) << std::endl;
            inputValues[input_channel][input_height][input_width] = batch.data[0][input_channel][input_height][input_width].item<float>();
            // std::cout << batch.data[0][input_channel][input_height][input_width].item<int>() << " : " << inputValues[input_channel][input_height][input_width] << std::endl;
          }
        }
      }
      // std::cout << batch.target[i].item<int64_t>() << " ";
      CreateBFloatInput(inputValues, inputExpValues, inputMantissaValues);
      int classifiedResult = ExecNeuralNetwork(inputExpValues, inputMantissaValues, weightExpVectors, weightMantissaVectors);
      std::cout << "target: " << targetResult << " classified: " << classifiedResult << std::endl;
      correctSample = (classifiedResult == targetResult) ? correctSample + 1 : correctSample;
    }
    std::cout << std::endl;
    sample++;
    if (sample >= maxSample){
      std::cout << "maxSample: " << maxSample << " correctSample: " << correctSample << std::endl;
      break;
    }
    // break;
  }
  #pragma endregion Load Input Data


  return 0;
}
