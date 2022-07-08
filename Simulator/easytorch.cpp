#include <cstdlib>
#include <iostream>
#include <math.h>

#include "PE.h"
#include "PEArray.h"
#include "Utils.h"

#include <torch/torch.h>
#include <torch/script.h>


void quantizeWeight(v<v<v<v<float>>>>& weightVector, v<v<v<v<int>>>>& quantizedWeightVector){
  float maxValue = 0;
  for (int output_channel = 0; output_channel < weightVector.size(); output_channel++){
    for (int input_channel = 0; input_channel < weightVector[0].size(); input_channel++){
      for (int kernel_height = 0; kernel_height < weightVector[0][0].size(); kernel_height++){
        for (int kernel_width = 0; kernel_width < weightVector[0][0][0].size(); kernel_width++){
          maxValue = std::max(maxValue, abs(weightVector[output_channel][input_channel][kernel_height][kernel_width]));
        }
      }
    }
  }

  for (int output_channel = 0; output_channel < weightVector.size(); output_channel++){
    for (int input_channel = 0; input_channel < weightVector[0].size(); input_channel++){
      for (int kernel_height = 0; kernel_height < weightVector[0][0].size(); kernel_height++){
        for (int kernel_width = 0; kernel_width < weightVector[0][0][0].size(); kernel_width++){
          quantizedWeightVector[output_channel][input_channel][kernel_height][kernel_width] = round(127 * weightVector[output_channel][input_channel][kernel_height][kernel_width] / maxValue);
          // std::cout << weightVector[output_channel][input_channel][kernel_height][kernel_width] << std::endl;
        }
      }
    }
  }
}

void quantizeInput(v<v<v<float>>>& inputValues, v<v<v<int>>>& quantizedInputValues){
  float maxValue = 0;
  for (int input_channel = 0; input_channel < inputValues.size(); input_channel++){
    for (int input_height = 0; input_height < inputValues[0].size(); input_height++){
      for (int input_width = 0; input_width < inputValues[0][0].size(); input_width++){
        maxValue = std::max(maxValue, abs(inputValues[input_channel][input_height][input_width]));
      }
    }
  }

  for (int input_channel = 0; input_channel < inputValues.size(); input_channel++){
    for (int input_height = 0; input_height < inputValues[0].size(); input_height++){
      for (int input_width = 0; input_width < inputValues[0][0].size(); input_width++){
        quantizedInputValues[input_channel][input_height][input_width] = round(127 * inputValues[input_channel][input_height][input_width] / maxValue);
      }
    }
  }
  return;
}

int ExecNeuralNetwork(v<v<v<int>>>& quantizedInputValues, v<v<v<v<v<int>>>>>& quantizedWeightVectors){
  v<v<v<int>>> layerInputValues;
  // initialize first layer input
  for (int input_channel = 0; input_channel < quantizedInputValues.size(); input_channel++){
    v<v<int>> temp1;
    for (int input_height = 0; input_height < quantizedInputValues[0].size(); input_height++){
      v<int> temp2;
      for (int input_width = 0; input_width < quantizedInputValues[0][0].size(); input_width++){
        temp2.push_back(quantizedInputValues[input_channel][input_height][input_width]);
      }
      temp1.push_back(temp2);
    }
    layerInputValues.push_back(temp1);
  }

  for (int layerIndex = 0; layerIndex < quantizedWeightVectors.size(); layerIndex++)
  {
    int num_output_channel = quantizedWeightVectors[layerIndex].size();
    int num_input_channel = quantizedWeightVectors[layerIndex][0].size();
    int num_kernel_height = quantizedWeightVectors[layerIndex][0][0].size();
    int num_kernel_width = quantizedWeightVectors[layerIndex][0][0][0].size();
    int num_input_height = layerInputValues[0].size();
    int num_input_width = layerInputValues[0][0].size();
    int stride = 1;

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    auto inputMemories = v<v<v<v<v<int>>>>>(simulator::num_PE_width, v<v<v<v<int>>>>(num_input_height, v<v<v<int>>>(num_input_width, v<v<int>>(num_input_channel, v<int>(simulator::num_PE_parallel)))));
    auto weightMemories = v<v<v<v<v<v<int>>>>>>(simulator::num_PE_height, v<v<v<v<v<int>>>>>(num_output_channel, v<v<v<v<int>>>>(num_kernel_height, v<v<v<int>>>(num_kernel_width, v<v<int>>(num_input_channel, v<int>(simulator::num_PE_parallel))))));
    auto outputValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));
    simulator::convertInputToInputMemoryFormat(layerInputValues, inputMemories, stride, num_kernel_height, num_kernel_width);
    simulator::convertWeightToWeightMemoryFormat(quantizedWeightVectors[layerIndex], weightMemories);

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

    // output is peArray.outputMemory
    layerInputValues = v<v<v<int>>>(num_output_channel, v<v<int>>(num_output_height, v<int>(num_output_width)));
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int output_height = 0; output_height < num_output_height; output_height++){
        for (int output_width = 0; output_width < num_output_width; output_width++){
          layerInputValues[output_channel][output_height][output_width] = peArray.outputMemory[output_channel][output_height][output_width];
          //layerInputValues[output_channel][output_height][output_width] = std::max(-127, std::min(127, peArray.outputMemory[output_channel][output_height][output_width]));
        }
      }
    }

    // quantize values
    int maxValue = 0;
    for (int input_channel = 0; input_channel < layerInputValues.size(); input_channel++){
      for (int input_height = 0; input_height < layerInputValues[0].size(); input_height++){
        for (int input_width = 0; input_width < layerInputValues[0][0].size(); input_width++){
          maxValue = std::max(maxValue, abs(layerInputValues[input_channel][input_height][input_width]));
        }
      }
    }
    for (int input_channel = 0; input_channel < layerInputValues.size(); input_channel++){
      for (int input_height = 0; input_height < layerInputValues[0].size(); input_height++){
        for (int input_width = 0; input_width < layerInputValues[0][0].size(); input_width++){
          layerInputValues[input_channel][input_height][input_width] = round(127 * layerInputValues[input_channel][input_height][input_width] / (float)maxValue);
        }
      }
    }

    // relu
    // relu is applied except the last layer
    if (layerIndex != quantizedWeightVectors.size() - 1){
      for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
        for (int output_height = 0; output_height < num_output_height; output_height++){
          for (int output_width = 0; output_width < num_output_width; output_width++){
            layerInputValues[output_channel][output_height][output_width] = std::max(layerInputValues[output_channel][output_height][output_width], 0);
          }
        }
      }
    }

    // x.view(-1)
    // view is applied only when layerIndex == 1, TODO: we need to update this part for generalization
    if (layerIndex == 1){
      auto newLayerInputValues = v<v<v<int>>>(num_output_channel*num_output_height*num_output_width, v<v<int>>(1, v<int>(1)));
      int viewIndex = -1;
      for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
        for (int output_height = 0; output_height < num_output_height; output_height++){
          for (int output_width = 0; output_width < num_output_width; output_width++){
            viewIndex++;
            newLayerInputValues[viewIndex][0][0] = layerInputValues[output_channel][output_height][output_width];
          }
        }
      }
      layerInputValues = newLayerInputValues;
    }

    std::cout << "layer: " << layerIndex << " cycle: " << cycles << std::endl;
  }

  std::cout << "output: " << layerInputValues << std::endl;
  int classifiedResult = std::max_element(layerInputValues.begin(),layerInputValues.end()) - layerInputValues.begin();
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
  std::vector<v<v<v<v<int>>>>> quantizedWeightVectors;

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

  // quantize weight
  for (int weightVectorIndex = 0; weightVectorIndex < weightVectors.size(); weightVectorIndex++){
    auto quantizedWeightVector = v<v<v<v<int>>>>(weightVectors[weightVectorIndex].size(), v<v<v<int>>>(weightVectors[weightVectorIndex][0].size(), v<v<int>>(weightVectors[weightVectorIndex][0][0].size(), v<int>(weightVectors[weightVectorIndex][0][0][0].size()))));
    quantizeWeight(weightVectors[weightVectorIndex], quantizedWeightVector);
    quantizedWeightVectors.push_back(quantizedWeightVector);
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
  auto quantizedInputValues = v<v<v<int>>>(num_input_channel, v<v<int>>(num_input_height, v<int>(num_input_width)));

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
      quantizeInput(inputValues, quantizedInputValues);
      int classifiedResult = ExecNeuralNetwork(quantizedInputValues, quantizedWeightVectors);
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
