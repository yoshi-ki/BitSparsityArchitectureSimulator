#include "PE.h"
#include "PEArray.h"
#include "BFloatPEArray.h"
#include <iostream>
#include <stdexcept> // std::runtime_error
#include "math.h"

namespace simulator
{

  void convertInputToInputMemoryFormat(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> &inputMemories,
    int stride,
    int num_kernel_height,
    int num_kernel_width
  )
  {
    int num_input_channel = inputValues.size();
    int num_input_height = inputValues[0].size();
    int num_input_width = inputValues[0][0].size();

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    int firstGroupOutputHeight = num_output_height / 2 + num_output_height % 2;
    int firstGroupOutputWidth = num_output_width / 2 + num_output_width % 2;

    for (int memoryIndex = 0; memoryIndex < num_PE_width; memoryIndex++){
      bool isheightFirstGroup = (memoryIndex / 2 == 0);
      bool iswidthFirstGroup = (memoryIndex % 2 == 0);

      int input_height_start = isheightFirstGroup ? 0 : firstGroupOutputHeight * stride;
      int input_height_end = isheightFirstGroup ? firstGroupOutputHeight * stride + num_kernel_height - 1: num_input_height;
      int input_width_start = iswidthFirstGroup ? 0 : firstGroupOutputWidth * stride;
      int input_width_end = iswidthFirstGroup ? firstGroupOutputWidth * stride + num_kernel_width - 1 : num_input_width;

      for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
        for (int input_height = input_height_start; input_height < input_height_end; input_height++){
          for (int input_width = input_width_start; input_width < input_width_end; input_width++){

            int height = input_height - input_height_start;
            int width = input_width - input_width_start;
            int channelGroup = input_channel / num_PE_parallel;
            inputMemories[memoryIndex][height][width][channelGroup][input_channel % num_PE_parallel] =
                inputValues[input_channel][input_height][input_width];
          }
        }
      }
    }
  };

  void convertWeightToWeightMemoryFormat(
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> &weightMemories
  )
  {
    int num_output_channel = weightValues.size();
    int num_input_channel = weightValues[0].size();
    int num_kernel_height = weightValues[0][0].size();
    int num_kernel_width = weightValues[0][0][0].size();
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
        for (int kernel_height = 0; kernel_height < num_kernel_height; kernel_height++){
          for (int kernel_width = 0; kernel_width < num_kernel_width; kernel_width++){
            int memoryIndex = output_channel % num_PE_height;
            int channelGroup = input_channel / num_PE_parallel;
            weightMemories[memoryIndex][output_channel / num_PE_height][kernel_height][kernel_width][channelGroup][input_channel % num_PE_parallel] =
                weightValues[output_channel][input_channel][kernel_height][kernel_width];
          }
        }
      }
    }
  };

  void computeConv(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<int>>> &outputValues,
    int stride
  )
  {
    int num_input_channel = inputValues.size();
    int num_input_height = inputValues[0].size();
    int num_input_width = inputValues[0][0].size();
    int num_output_channel = weightValues.size();
    int num_kernel_height = weightValues[0][0].size();
    int num_kernel_width = weightValues[0][0][0].size();

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int output_height = 0; output_height < num_output_height; output_height++){
        for (int output_width = 0; output_width < num_output_width; output_width++){
          // when we want to write output_channel, output_height, output_width
          int outputValue = 0;
          for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
            for (int kernel_height = 0; kernel_height < num_kernel_height; kernel_height++){
              for (int kernel_width = 0; kernel_width < num_kernel_width; kernel_width++){
                outputValue +=
                  (inputValues[input_channel][output_height*stride + kernel_height][output_width*stride + kernel_width] *
                    weightValues[output_channel][input_channel][kernel_height][kernel_width]);
              }
            }
          }

          outputValues[output_channel][output_height][output_width] = outputValue;
        }
      }
    }
  };

  void computeConvFloat(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<int>>> &inputExpValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightExpValues,
    std::vector<std::vector<std::vector<float>>> &outputValues,
    int stride
  )
  {
    int num_input_channel = inputValues.size();
    int num_input_height = inputValues[0].size();
    int num_input_width = inputValues[0][0].size();
    int num_output_channel = weightValues.size();
    int num_kernel_height = weightValues[0][0].size();
    int num_kernel_width = weightValues[0][0][0].size();

    int num_output_height = ((num_input_height - num_kernel_height) / stride) + 1;
    int num_output_width = ((num_input_width - num_kernel_width) / stride) + 1;

    auto inputFloatValues = v<v<v<float>>>(num_input_channel, v<v<float>>(num_input_height, v<float>(num_input_width)));
    auto weightFloatValues = v<v<v<v<float>>>>(num_output_channel, v<v<v<float>>>(num_input_channel, v<v<float>>(num_kernel_height, v<float>(num_kernel_width))));
    for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
      for (int input_height = 0; input_height < num_input_height; input_height++){
        for (int input_width = 0; input_width < num_input_width; input_width++){
          inputFloatValues[input_channel][input_height][input_width] = CreateFloatFromBFloat(
            std::make_pair(inputExpValues[input_channel][input_height][input_width], inputValues[input_channel][input_height][input_width])
          );
          // std::cout << "create float from bfloat: " << inputExpValues[input_channel][input_height][input_width] << " " << inputValues[input_channel][input_height][input_width] << std::endl;
          // std::cout << inputFloatValues[input_channel][input_height][input_width] << std::endl;
        }
      }
    }
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
        for (int kernel_height = 0; kernel_height < num_kernel_height; kernel_height++){
          for (int kernel_width = 0; kernel_width < num_kernel_width; kernel_width++){
            weightFloatValues[output_channel][input_channel][kernel_height][kernel_width] = CreateFloatFromBFloat(
              std::make_pair(weightExpValues[output_channel][input_channel][kernel_height][kernel_width], weightValues[output_channel][input_channel][kernel_height][kernel_width])
            );
          }
        }
      }
    }

    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      for (int output_height = 0; output_height < num_output_height; output_height++){
        for (int output_width = 0; output_width < num_output_width; output_width++){
          // when we want to write output_channel, output_height, output_width
          float outputValue = 0;
          for (int input_channel = 0; input_channel < num_input_channel; input_channel++){
            for (int kernel_height = 0; kernel_height < num_kernel_height; kernel_height++){
              for (int kernel_width = 0; kernel_width < num_kernel_width; kernel_width++){
                // std::cout << "input: " << inputFloatValues[input_channel][output_height*stride + kernel_height][output_width*stride + kernel_width] << " weight: " << weightFloatValues[output_channel][input_channel][kernel_height][kernel_width] << std::endl;
                outputValue +=
                  (inputFloatValues[input_channel][output_height*stride + kernel_height][output_width*stride + kernel_width] *
                    weightFloatValues[output_channel][input_channel][kernel_height][kernel_width]);
              }
            }
          }

          outputValues[output_channel][output_height][output_width] = outputValue;
        }
      }
    }
  };

  #pragma region PEArrayUtils

  void convertInputMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
    int num_input_channel,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    int iterationForOutputChannelGroup = num_output_channel / num_PE_height + fmin(num_output_channel % num_PE_height, 1);
    int num_input_channel_group = num_input_channel / num_PE_parallel + fmin(num_input_channel % num_PE_parallel, 1);
    // we need to set the same values to fifo for different output channel groups
    for (int iter = 0; iter < iterationForOutputChannelGroup; iter++){
      for (int memoryIndex = 0; memoryIndex < inputMemories.size(); memoryIndex++){
        // input is divided into four areas, so we need to compute the size of the each areas

        int num_output_height = ((input_height - kernel_height) / stride) + 1;
        int num_output_width = ((input_width - kernel_width) / stride) + 1;
        int firstGroupOutputHeight = num_output_height / 2 + num_output_height % 2;
        int firstGroupOutputWidth = num_output_width / 2 + num_output_width % 2;
        int partialInputHeight = (memoryIndex == 0 || memoryIndex == 1) ? firstGroupOutputHeight : num_output_height - firstGroupOutputHeight;
        int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? firstGroupOutputWidth : num_output_width - firstGroupOutputWidth;

        // start position of window
        for (int windowStartHeight = 0; windowStartHeight < partialInputHeight; windowStartHeight = windowStartHeight + stride){
          for (int windowStartWidth = 0; windowStartWidth < partialInputWidth; windowStartWidth = windowStartWidth + stride){
            for (int kh = 0; kh < kernel_height; kh++){
              for (int kw = 0; kw < kernel_width; kw++){
                for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
                  // input
                  for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                    if (channelGroup == num_input_channel_group - 1 && channelGroup * num_PE_parallel + input_channel >= num_input_channel){
                      // when last group and there are no values
                      continue;
                    }
                    inputValuesFifos[memoryIndex][input_channel].push_back(
                        FIFOValues{
                            inputMemories[memoryIndex][windowStartHeight + kh][windowStartWidth + kw][channelGroup][input_channel],
                            false});
                  }
                }
              }
            }
            // insert end of the psum
            for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
              if(inputValuesFifos[memoryIndex][input_channel].empty()){
                continue;
              }

              auto lastVal = inputValuesFifos[memoryIndex][input_channel].back();
              lastVal.isLast= true;
              inputValuesFifos[memoryIndex][input_channel].pop_back();
              inputValuesFifos[memoryIndex][input_channel].push_back(lastVal);
            }
          }
        }
      }
    }
  }

  void convertInputMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputExpMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
    std::vector<std::vector<std::deque<int>>>& inputExpFifos,
    int num_input_channel,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    int iterationForOutputChannelGroup = num_output_channel / num_PE_height + fmin(num_output_channel % num_PE_height, 1);
    int num_input_channel_group = num_input_channel / num_PE_parallel + fmin(num_input_channel % num_PE_parallel, 1);
    // we need to set the same values to fifo for different output channel groups
    for (int iter = 0; iter < iterationForOutputChannelGroup; iter++){
      for (int memoryIndex = 0; memoryIndex < inputMemories.size(); memoryIndex++){
        // input is divided into four areas, so we need to compute the size of the each areas

        int num_output_height = ((input_height - kernel_height) / stride) + 1;
        int num_output_width = ((input_width - kernel_width) / stride) + 1;
        int firstGroupOutputHeight = num_output_height / 2 + num_output_height % 2;
        int firstGroupOutputWidth = num_output_width / 2 + num_output_width % 2;
        int partialInputHeight = (memoryIndex == 0 || memoryIndex == 1) ? firstGroupOutputHeight : num_output_height - firstGroupOutputHeight;
        int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? firstGroupOutputWidth : num_output_width - firstGroupOutputWidth;

        // start position of window
        for (int windowStartHeight = 0; windowStartHeight < partialInputHeight; windowStartHeight = windowStartHeight + stride){
          for (int windowStartWidth = 0; windowStartWidth < partialInputWidth; windowStartWidth = windowStartWidth + stride){
            for (int kh = 0; kh < kernel_height; kh++){
              for (int kw = 0; kw < kernel_width; kw++){
                for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
                  // input
                  for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                    if (channelGroup == num_input_channel_group - 1 && channelGroup * num_PE_parallel + input_channel >= num_input_channel){
                      // when last group and there are no values
                      continue;
                    }
                    inputValuesFifos[memoryIndex][input_channel].push_back(
                        FIFOValues{
                            inputMemories[memoryIndex][windowStartHeight + kh][windowStartWidth + kw][channelGroup][input_channel],
                            false});
                    // if (input_channel == 0){
                    //   std::cout << memoryIndex << " " << windowStartHeight + kh << " " << windowStartWidth + kw << " " << channelGroup << std::endl;
                    //   std::cout << inputExpMemories[memoryIndex][windowStartHeight + kh][windowStartWidth + kw][channelGroup][input_channel] << std::endl;
                    // }
                    inputExpFifos[memoryIndex][input_channel].push_back(inputExpMemories[memoryIndex][windowStartHeight + kh][windowStartWidth + kw][channelGroup][input_channel]);
                  }
                }
              }
            }
            // insert end of the psum
            for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
              if(inputValuesFifos[memoryIndex][input_channel].empty()){
                continue;
              }

              auto lastVal = inputValuesFifos[memoryIndex][input_channel].back();
              lastVal.isLast= true;
              inputValuesFifos[memoryIndex][input_channel].pop_back();
              inputValuesFifos[memoryIndex][input_channel].push_back(lastVal);
            }
          }
        }
      }
    }
  }

  void convertWeightMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
    int num_input_channel,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    int num_input_channel_group = num_input_channel / num_PE_parallel + fmin(num_input_channel % num_PE_parallel, 1);
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      int memoryIndex = output_channel % num_PE_height;

      int num_output_height = ((input_height - kernel_height) / stride) + 1;
      int num_output_width = ((input_width - kernel_width) / stride) + 1;
      int firstGroupOutputHeight = num_output_height / 2 + num_output_height % 2;
      int firstGroupOutputWidth = num_output_width / 2 + num_output_width % 2;
      int partialInputHeight = firstGroupOutputHeight; // use bigger height
      int partialInputWidth = firstGroupOutputWidth; // use bigger width

      // this is not the start position here, it is just iteration
      for (int windowStartHeight = 0; windowStartHeight < partialInputHeight; windowStartHeight = windowStartHeight + stride){
        for (int windowStartWidth = 0; windowStartWidth < partialInputWidth; windowStartWidth = windowStartWidth + stride){

          for (int kh = 0; kh < kernel_height; kh++){
            for (int kw = 0; kw < kernel_width; kw++){
              for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
                // input
                for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                  if (channelGroup == num_input_channel_group - 1 && channelGroup * num_PE_parallel + input_channel >= num_input_channel){
                    // when last group and there are no values
                    continue;
                  }
                  weightValuesFifos[memoryIndex][input_channel].push_back(
                    FIFOValues{
                        weightMemories[memoryIndex][output_channel / num_PE_height][kh][kw][channelGroup][input_channel],
                        false
                    });
                }
              }
            }
          }
          // insert end of the psum
          for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
            if(weightValuesFifos[memoryIndex][input_channel].empty()){
              continue;
            }
            auto lastVal = weightValuesFifos[memoryIndex][input_channel].back();
            lastVal.isLast = true;
            weightValuesFifos[memoryIndex][input_channel].pop_back();
            weightValuesFifos[memoryIndex][input_channel].push_back(lastVal);
          }
        }
      }
    }
  };

  void convertWeightMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightExpMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
    std::vector<std::vector<std::deque<int>>>& weightExpFifos,
    int num_input_channel,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    int num_input_channel_group = num_input_channel / num_PE_parallel + fmin(num_input_channel % num_PE_parallel, 1);
    for (int output_channel = 0; output_channel < num_output_channel; output_channel++){
      int memoryIndex = output_channel % num_PE_height;

      int num_output_height = ((input_height - kernel_height) / stride) + 1;
      int num_output_width = ((input_width - kernel_width) / stride) + 1;
      int firstGroupOutputHeight = num_output_height / 2 + num_output_height % 2;
      int firstGroupOutputWidth = num_output_width / 2 + num_output_width % 2;
      int partialInputHeight = firstGroupOutputHeight; // use bigger height
      int partialInputWidth = firstGroupOutputWidth; // use bigger width

      // this is not the start position here, it is just iteration
      for (int windowStartHeight = 0; windowStartHeight < partialInputHeight; windowStartHeight = windowStartHeight + stride){
        for (int windowStartWidth = 0; windowStartWidth < partialInputWidth; windowStartWidth = windowStartWidth + stride){

          for (int kh = 0; kh < kernel_height; kh++){
            for (int kw = 0; kw < kernel_width; kw++){
              for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
                // input
                for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                  if (channelGroup == num_input_channel_group - 1 && channelGroup * num_PE_parallel + input_channel >= num_input_channel){
                    // when last group and there are no values
                    continue;
                  }
                  weightValuesFifos[memoryIndex][input_channel].push_back(
                    FIFOValues{
                        weightMemories[memoryIndex][output_channel / num_PE_height][kh][kw][channelGroup][input_channel],
                        false
                    });
                  weightExpFifos[memoryIndex][input_channel].push_back(weightExpMemories[memoryIndex][output_channel / num_PE_height][kh][kw][channelGroup][input_channel]);
                }
              }
            }
          }
          // insert end of the psum
          for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
            if(weightValuesFifos[memoryIndex][input_channel].empty()){
              continue;
            }
            auto lastVal = weightValuesFifos[memoryIndex][input_channel].back();
            lastVal.isLast = true;
            weightValuesFifos[memoryIndex][input_channel].pop_back();
            weightValuesFifos[memoryIndex][input_channel].push_back(lastVal);
          }
        }
      }
    }
  };

  bool isLayerFinished(
    std::vector<std::vector<std::deque<FIFOValues>>>& inputFifos,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightFifos
  )
  {
    bool isFinished = true;
    for (int i = 0; i < num_PE_width; i++){
      for (int j = 0; j < num_PE_parallel; j++){
        isFinished = isFinished && inputFifos[i][j].empty();
      }
    }
    for (int i = 0; i < num_PE_height; i++){
      for (int j = 0; j < num_PE_parallel; j++){
        isFinished = isFinished && weightFifos[i][j].empty();
      }
    }
    return isFinished;
  };

  bool isFinishedPSumExecution(
    std::vector<PEControllerStatus>& inputControllerStatusForPEs,
    std::vector<PEControllerStatus>& weightControllerStatusForPEs
  )
  {
    // if all psum finished equal true
    bool isFinished = true;
    for (int i = 0; i < num_PE_width; i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        isFinished = isFinished && inputControllerStatusForPEs[i].finishedPSum[bitIndex];
      }
    }
    for (int i = 0; i < num_PE_height; i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        isFinished = isFinished && weightControllerStatusForPEs[i].finishedPSum[bitIndex];
      }
    }
    return isFinished;
  };

  void decodeValuesToBits(
    std::vector<std::vector<std::deque<FIFOValues>>>& valueFifos,
    std::vector<DecodedRegister>& decodedRepresentations
  )
  {
    for (int fifoIndex = 0; fifoIndex < valueFifos.size(); fifoIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        if (valueFifos[fifoIndex][input_channel].size() == 0){
          decodedRepresentations[fifoIndex].isValids[input_channel][0] = false;
          continue;
        }

        int val = valueFifos[fifoIndex][input_channel].front().value;

        // negative values transformation. considering future use, we set the isNegatives as vector
        if (val >= 0){
          for (int i = 0; i < 8; i++){
            decodedRepresentations[fifoIndex].isNegatives[input_channel][i] = false;
          }
        }
        else{
          for (int i = 0; i < 8; i++){
            decodedRepresentations[fifoIndex].isNegatives[input_channel][i] = true;
          }
          val = -val;
        }

        int bitVectorIndex = 0;
        // i == 7 is sign bit for our case
        for (int i = 6; i >= 0; i--)
        {
          int mask = 1 << i;
          if ((val & mask) > 0)
          {
            decodedRepresentations[fifoIndex].bitInputValues[input_channel][bitVectorIndex] = (unsigned int)(i);
            decodedRepresentations[fifoIndex].isValids[input_channel][bitVectorIndex] = true;
            bitVectorIndex++;
          }
        }
      }
    }
  };

  void decodeValuesToBitsWithLeadingOne(
    std::vector<std::vector<std::deque<FIFOValues>>>& valueFifos,
    std::vector<std::vector<std::deque<int>>>& expFifos,
    std::vector<DecodedRegister>& decodedRepresentations
  )
  {
    for (int fifoIndex = 0; fifoIndex < valueFifos.size(); fifoIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        if (valueFifos[fifoIndex][input_channel].size() == 0){
          decodedRepresentations[fifoIndex].isValids[input_channel][0] = false;
          continue;
        }

        int val = valueFifos[fifoIndex][input_channel].front().value;
        int exp = expFifos[fifoIndex][input_channel].front();

        // negative values transformation. considering future use, we set the isNegatives as vector
        if (val < 128){
          for (int i = 0; i < 8; i++){
            decodedRepresentations[fifoIndex].isNegatives[input_channel][i] = false;
          }
        }
        else{
          for (int i = 0; i < 8; i++){
            decodedRepresentations[fifoIndex].isNegatives[input_channel][i] = true;
          }
          val = -128;
        }

        int bitVectorIndex = 0;
        // i == 7 is 1 for mantissa in bfloat.
        if (exp != 0 || val != 0){
          decodedRepresentations[fifoIndex].bitInputValues[input_channel][bitVectorIndex] = (unsigned int)(7);
          decodedRepresentations[fifoIndex].isValids[input_channel][bitVectorIndex] = true;
          bitVectorIndex++;
        }

        for (int i = 6; i >= 0; i--)
        {
          int mask = 1 << i;
          if ((val & mask) > 0)
          {
            decodedRepresentations[fifoIndex].bitInputValues[input_channel][bitVectorIndex] = (unsigned int)(i);
            decodedRepresentations[fifoIndex].isValids[input_channel][bitVectorIndex] = true;
            bitVectorIndex++;
          }
        }

        // if (input_channel == 0){std::cout << "exp: " << exp << " val: " << val << std::endl;}

        for (int i = bitVectorIndex; i < 8; i++){
          decodedRepresentations[fifoIndex].isValids[input_channel][bitVectorIndex] = false;
        }
      }
    }
  };

  void createInputForPEsBasedOnControllerStatus(
    std::vector<DecodedRegister>& decodedRegisters,
    std::vector<PEControllerStatus>& controllerStatusForPEs,
    std::vector<PEInput>& representationsForPEs,
    int num_Fifo
  )
  {
    for (int fifoIndex = 0; fifoIndex < num_Fifo; fifoIndex++){
      auto controllerStatus = controllerStatusForPEs[fifoIndex];
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        if(controllerStatus.isWaiting[bitIndex]){
          // when waiting for next value
          representationsForPEs[fifoIndex].isValid[bitIndex] = false;
        }
        else{
          int processIndex = controllerStatus.nextProcessIndex[bitIndex];
          representationsForPEs[fifoIndex].bitInputValue[bitIndex] = decodedRegisters[fifoIndex].bitInputValues[bitIndex][processIndex];
          representationsForPEs[fifoIndex].isNegative[bitIndex] = decodedRegisters[fifoIndex].isNegatives[bitIndex][processIndex];
          representationsForPEs[fifoIndex].isValid[bitIndex] = decodedRegisters[fifoIndex].isValids[bitIndex][processIndex];
        }
      }
    }
  };

  void updatePEStatus(
    std::vector<PEControllerStatus>& inputControllerStatusForPEs,
    std::vector<PEControllerStatus>& weightControllerStatusForPEs,
    std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
    std::vector<DecodedRegister>& decodedInputs,
    std::vector<DecodedRegister>& decodedWeights
  )
  {
    auto newInputControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_width);
    auto newWeightControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_height);

    auto tempInputIsWaiting = std::vector<std::vector<bool>>(num_PE_width, v<bool>(num_PE_parallel));
    // Stage 1: create new input controller status only by the last input controller status
    for (int fifoIndex = 0; fifoIndex < num_PE_width; fifoIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        int nowProcessIndex = inputControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex];
        bool isValidNow = decodedInputs[fifoIndex].isValids[bitIndex][nowProcessIndex];

        // if the processed valus is valid, we should take a look at the next value
        int nextProcessIndex = isValidNow ? nowProcessIndex + 1 : nowProcessIndex;
        newInputControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] = nextProcessIndex;
        newInputControllerStatusForPEs[fifoIndex].isWaiting[bitIndex] = !decodedInputs[fifoIndex].isValids[bitIndex][nextProcessIndex];
        tempInputIsWaiting[fifoIndex][bitIndex] = !decodedInputs[fifoIndex].isValids[bitIndex][nextProcessIndex];

        // we do not change the status for finishedPsum here. when no weight is set, we have to explicitly set true for preventing infinite loop
        if (decodedInputs[fifoIndex].isValids[bitIndex][0] == false){
          newInputControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex] = true;
        }
        else{
          newInputControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex] = inputControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex];
        }
      }
    }

    auto tempWeightIsWaiting = std::vector<std::vector<bool>>(num_PE_height, v<bool>(num_PE_parallel));
    // Stage 2: create new weight controller status
    for (int fifoIndex = 0; fifoIndex < num_PE_height; fifoIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        // decide we should consume next bit values. we should consume next bit when all the num_PE_width input fifos are waiting.
        bool weightForThisBitNext = true;
        for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++)
        {
          weightForThisBitNext = weightForThisBitNext && tempInputIsWaiting[inputFifoIndex][bitIndex];
        }

        if (weightForThisBitNext){
          // we will consume weight next bit from the next cycle
          int nowProcessIndex = weightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex];
          bool isValidNow = decodedWeights[fifoIndex].isValids[bitIndex][nowProcessIndex];
          int nextProcessIndex = isValidNow ? nowProcessIndex + 1 : nowProcessIndex;
          newWeightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] = nextProcessIndex;
          // std::cout << "nextProcessIndex" << nextProcessIndex << std::endl;

          bool weightFifoWaiting = !decodedWeights[fifoIndex].isValids[bitIndex][nextProcessIndex];
          newWeightControllerStatusForPEs[fifoIndex].isWaiting[bitIndex] = weightFifoWaiting;
          tempWeightIsWaiting[fifoIndex][bitIndex] = weightFifoWaiting;

          // we need to update the status for input controller if new weight bit is produced
          if (!weightFifoWaiting){
            for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++)
            {
              // have to overwrite the status for input
              newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = false;
              newInputControllerStatusForPEs[inputFifoIndex].nextProcessIndex[bitIndex] = 0;
            }
          }
        }
        else{
          newWeightControllerStatusForPEs[fifoIndex].isWaiting[bitIndex] = weightControllerStatusForPEs[fifoIndex].isWaiting[bitIndex];
          newWeightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] = weightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex];
        }

        if(decodedWeights[fifoIndex].isValids[bitIndex][0] == false){
          newWeightControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex] = true;
        }
        else{
          newWeightControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex] = weightControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex];
        }
      }
    }

    // Stage 3: decide we will update the fifo or not
    for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
      // we consume next value for input and weight when all weight FIFO is waiting for next values
      bool updateFifo = true;
      for (int weightFifoIndex = 0; weightFifoIndex < num_PE_height; weightFifoIndex++)
      {
        updateFifo = updateFifo && tempWeightIsWaiting[weightFifoIndex][bitIndex];
      }

      // update for activation and weight is done at the same time
      if(updateFifo){
        // update input FIFO
        for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++){
          if (!inputValuesFifos[inputFifoIndex][bitIndex].empty()){
            // if the value is the last values of psum, we need to set wait status
            if (inputValuesFifos[inputFifoIndex][bitIndex].front().isLast)
            {
              newInputControllerStatusForPEs[inputFifoIndex].finishedPSum[bitIndex] = true;
              newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = true;
            }
            else{
              inputValuesFifos[inputFifoIndex][bitIndex].pop_front();
              newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = false;
              newInputControllerStatusForPEs[inputFifoIndex].nextProcessIndex[bitIndex] = 0;
              newInputControllerStatusForPEs[inputFifoIndex].finishedPSum[bitIndex] = false;
            }
          }

          // we consume next bit from the next cycle, so we will reset the decoded status too
          // this is done here to accelerate the simulator
          for (int i = 0; i < 8; i++){
            decodedInputs[inputFifoIndex].isValids[bitIndex][i] = false;
          }
        }

        // update weight FIFO
        for (int weightFifoIndex = 0; weightFifoIndex < num_PE_height; weightFifoIndex++)
        {
          if (!weightValuesFifos[weightFifoIndex][bitIndex].empty()){
            // if the value is the last values of psum, we need to set wait status
            if (weightValuesFifos[weightFifoIndex][bitIndex].front().isLast)
            {
              newWeightControllerStatusForPEs[weightFifoIndex].finishedPSum[bitIndex] = true;
              newWeightControllerStatusForPEs[weightFifoIndex].isWaiting[bitIndex] = true;
            }
            else{
              weightValuesFifos[weightFifoIndex][bitIndex].pop_front();
              newWeightControllerStatusForPEs[weightFifoIndex].isWaiting[bitIndex] = false;
              newWeightControllerStatusForPEs[weightFifoIndex].nextProcessIndex[bitIndex] = 0;
              newWeightControllerStatusForPEs[weightFifoIndex].finishedPSum[bitIndex] = false;
            }
          }

          // we consume next bit from the next cycle, so we will reset the decoded status too
          // this is done here to accelerate the simulator
          for (int i = 0; i < 8; i++){
            decodedWeights[weightFifoIndex].isValids[bitIndex][i] = false;
          }
        }
      }
    }

    inputControllerStatusForPEs = newInputControllerStatusForPEs;
    weightControllerStatusForPEs = newWeightControllerStatusForPEs;

    return ;
  };

  void updatePEStatus(
    std::vector<PEControllerStatus>& inputControllerStatusForPEs,
    std::vector<PEControllerStatus>& weightControllerStatusForPEs,
    std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
    std::vector<std::vector<std::deque<int>>>& inputExpFifos,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
    std::vector<std::vector<std::deque<int>>>& weightExpFifos,
    std::vector<DecodedRegister>& decodedInputs,
    std::vector<DecodedRegister>& decodedWeights
  )
  {
    auto newInputControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_width);
    auto newWeightControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_height);

    auto tempInputIsWaiting = std::vector<std::vector<bool>>(num_PE_width, v<bool>(num_PE_parallel));
    // Stage 1: create new input controller status only by the last input controller status
    for (int fifoIndex = 0; fifoIndex < num_PE_width; fifoIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        int nowProcessIndex = inputControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex];
        bool isValidNow = decodedInputs[fifoIndex].isValids[bitIndex][nowProcessIndex];

        // if the processed valus is valid, we should take a look at the next value
        int nextProcessIndex = isValidNow ? nowProcessIndex + 1 : nowProcessIndex;
        newInputControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] = nextProcessIndex;
        newInputControllerStatusForPEs[fifoIndex].isWaiting[bitIndex] = !decodedInputs[fifoIndex].isValids[bitIndex][nextProcessIndex];
        tempInputIsWaiting[fifoIndex][bitIndex] = !decodedInputs[fifoIndex].isValids[bitIndex][nextProcessIndex];

        // we do not change the status for finishedPsum here. when no weight is set, we have to explicitly set true for preventing infinite loop
        if (decodedInputs[fifoIndex].isValids[bitIndex][0] == false){
          newInputControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex] = true;
        }
        else{
          newInputControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex] = inputControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex];
        }
      }
    }

    auto tempWeightIsWaiting = std::vector<std::vector<bool>>(num_PE_height, v<bool>(num_PE_parallel));
    // Stage 2: create new weight controller status
    for (int fifoIndex = 0; fifoIndex < num_PE_height; fifoIndex++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        // decide we should consume next bit values. we should consume next bit when all the num_PE_width input fifos are waiting.
        bool weightForThisBitNext = true;
        for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++)
        {
          weightForThisBitNext = weightForThisBitNext && tempInputIsWaiting[inputFifoIndex][bitIndex];
        }

        if (weightForThisBitNext){
          // we will consume weight next bit from the next cycle
          int nowProcessIndex = weightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex];
          bool isValidNow = decodedWeights[fifoIndex].isValids[bitIndex][nowProcessIndex];
          int nextProcessIndex = isValidNow ? nowProcessIndex + 1 : nowProcessIndex;
          newWeightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] = nextProcessIndex;
          // std::cout << "nextProcessIndex" << nextProcessIndex << std::endl;

          bool weightFifoWaiting = !decodedWeights[fifoIndex].isValids[bitIndex][nextProcessIndex];
          newWeightControllerStatusForPEs[fifoIndex].isWaiting[bitIndex] = weightFifoWaiting;
          tempWeightIsWaiting[fifoIndex][bitIndex] = weightFifoWaiting;

          // we need to update the status for input controller if new weight bit is produced
          if (!weightFifoWaiting){
            for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++)
            {
              // have to overwrite the status for input
              newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = false;
              newInputControllerStatusForPEs[inputFifoIndex].nextProcessIndex[bitIndex] = 0;
            }
          }
        }
        else{
          newWeightControllerStatusForPEs[fifoIndex].isWaiting[bitIndex] = weightControllerStatusForPEs[fifoIndex].isWaiting[bitIndex];
          newWeightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex] = weightControllerStatusForPEs[fifoIndex].nextProcessIndex[bitIndex];
        }

        if(decodedWeights[fifoIndex].isValids[bitIndex][0] == false){
          newWeightControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex] = true;
        }
        else{
          newWeightControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex] = weightControllerStatusForPEs[fifoIndex].finishedPSum[bitIndex];
        }
      }
    }

    // Stage 3: decide we will update the fifo or not
    for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
      // we consume next value for input and weight when all weight FIFO is waiting for next values
      bool updateFifo = true;
      for (int weightFifoIndex = 0; weightFifoIndex < num_PE_height; weightFifoIndex++)
      {
        updateFifo = updateFifo && tempWeightIsWaiting[weightFifoIndex][bitIndex];
      }

      // update for activation and weight is done at the same time
      if(updateFifo){
        // update input FIFO
        for (int inputFifoIndex = 0; inputFifoIndex < num_PE_width; inputFifoIndex++){
          if (!inputValuesFifos[inputFifoIndex][bitIndex].empty()){
            // if the value is the last values of psum, we need to set wait status
            if (inputValuesFifos[inputFifoIndex][bitIndex].front().isLast)
            {
              newInputControllerStatusForPEs[inputFifoIndex].finishedPSum[bitIndex] = true;
              newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = true;
            }
            else{
              inputValuesFifos[inputFifoIndex][bitIndex].pop_front();
              inputExpFifos[inputFifoIndex][bitIndex].pop_front();
              newInputControllerStatusForPEs[inputFifoIndex].isWaiting[bitIndex] = false;
              newInputControllerStatusForPEs[inputFifoIndex].nextProcessIndex[bitIndex] = 0;
              newInputControllerStatusForPEs[inputFifoIndex].finishedPSum[bitIndex] = false;
            }
          }

          // we consume next bit from the next cycle, so we will reset the decoded status too
          // this is done here to accelerate the simulator
          for (int i = 0; i < 8; i++){
            decodedInputs[inputFifoIndex].isValids[bitIndex][i] = false;
          }
        }

        // update weight FIFO
        for (int weightFifoIndex = 0; weightFifoIndex < num_PE_height; weightFifoIndex++)
        {
          if (!weightValuesFifos[weightFifoIndex][bitIndex].empty()){
            // if the value is the last values of psum, we need to set wait status
            if (weightValuesFifos[weightFifoIndex][bitIndex].front().isLast)
            {
              newWeightControllerStatusForPEs[weightFifoIndex].finishedPSum[bitIndex] = true;
              newWeightControllerStatusForPEs[weightFifoIndex].isWaiting[bitIndex] = true;
            }
            else{
              weightValuesFifos[weightFifoIndex][bitIndex].pop_front();
              weightExpFifos[weightFifoIndex][bitIndex].pop_front();
              newWeightControllerStatusForPEs[weightFifoIndex].isWaiting[bitIndex] = false;
              newWeightControllerStatusForPEs[weightFifoIndex].nextProcessIndex[bitIndex] = 0;
              newWeightControllerStatusForPEs[weightFifoIndex].finishedPSum[bitIndex] = false;
            }
          }

          // we consume next bit from the next cycle, so we will reset the decoded status too
          // this is done here to accelerate the simulator
          for (int i = 0; i < 8; i++){
            decodedWeights[weightFifoIndex].isValids[bitIndex][i] = false;
          }
        }
      }
    }

    inputControllerStatusForPEs = newInputControllerStatusForPEs;
    weightControllerStatusForPEs = newWeightControllerStatusForPEs;

    return ;
  };

  void updatePEStatusWhenPsumFinish(
    std::vector<PEControllerStatus> & inputControllerStatusForPEs,
    std::vector<PEControllerStatus> & weightControllerStatusForPEs,
    std::vector<std::vector<std::deque<FIFOValues>>> & inputValuesFifos,
    std::vector<std::vector<std::deque<FIFOValues>>> & weightValuesFifos
  )
  {
    // finishedPSum -> false, isWaiting -> false, index -> 0, pop fifo
    for (int i = 0; i < inputControllerStatusForPEs.size(); i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        inputControllerStatusForPEs[i].finishedPSum[bitIndex] = false;
        inputControllerStatusForPEs[i].isWaiting[bitIndex] = false;
        inputControllerStatusForPEs[i].nextProcessIndex[bitIndex] = 0;
      }
    }
    for (int memoryIndex = 0; memoryIndex < inputValuesFifos.size(); memoryIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        if (inputValuesFifos[memoryIndex][input_channel].size() != 0){
          inputValuesFifos[memoryIndex][input_channel].pop_front();
        }
      }
    }

    for (int i = 0; i < weightControllerStatusForPEs.size(); i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        weightControllerStatusForPEs[i].finishedPSum[bitIndex] = false;
        weightControllerStatusForPEs[i].isWaiting[bitIndex] = false;
        weightControllerStatusForPEs[i].nextProcessIndex[bitIndex] = 0;
      }
    }
    for (int memoryIndex = 0; memoryIndex < weightValuesFifos.size(); memoryIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        if(weightValuesFifos[memoryIndex][input_channel].size() != 0){
          weightValuesFifos[memoryIndex][input_channel].pop_front();
        }
      }
    }
  };

  void updatePEStatusWhenPsumFinish(
    std::vector<PEControllerStatus> & inputControllerStatusForPEs,
    std::vector<PEControllerStatus> & weightControllerStatusForPEs,
    std::vector<std::vector<std::deque<FIFOValues>>> & inputValuesFifos,
    std::vector<std::vector<std::deque<int>>> & inputExpFifos,
    std::vector<std::vector<std::deque<FIFOValues>>> & weightValuesFifos,
    std::vector<std::vector<std::deque<int>>> & weightExpFifos
  )
  {
    // finishedPSum -> false, isWaiting -> false, index -> 0, pop fifo
    for (int i = 0; i < inputControllerStatusForPEs.size(); i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        inputControllerStatusForPEs[i].finishedPSum[bitIndex] = false;
        inputControllerStatusForPEs[i].isWaiting[bitIndex] = false;
        inputControllerStatusForPEs[i].nextProcessIndex[bitIndex] = 0;
      }
    }
    for (int memoryIndex = 0; memoryIndex < inputValuesFifos.size(); memoryIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        if (inputValuesFifos[memoryIndex][input_channel].size() != 0){
          inputValuesFifos[memoryIndex][input_channel].pop_front();
          inputExpFifos[memoryIndex][input_channel].pop_front();
        }
      }
    }

    for (int i = 0; i < weightControllerStatusForPEs.size(); i++){
      for (int bitIndex = 0; bitIndex < num_PE_parallel; bitIndex++){
        weightControllerStatusForPEs[i].finishedPSum[bitIndex] = false;
        weightControllerStatusForPEs[i].isWaiting[bitIndex] = false;
        weightControllerStatusForPEs[i].nextProcessIndex[bitIndex] = 0;
      }
    }
    for (int memoryIndex = 0; memoryIndex < weightValuesFifos.size(); memoryIndex++){
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        if(weightValuesFifos[memoryIndex][input_channel].size() != 0){
          weightValuesFifos[memoryIndex][input_channel].pop_front();
          weightExpFifos[memoryIndex][input_channel].pop_front();
        }
      }
    }
  };

  void writeOutput(
    std::vector<std::vector<int>>& outputOfPEs,
    std::vector<std::vector<std::vector<int>>>& outputMemory,
    int outputStatus, // index of output, when first output, it is zero
    int output_height,
    int output_width,
    int num_output_channel
  )
  {
    for (int h = 0; h < num_PE_height; h++)
    {
      for (int w = 0; w < num_PE_width; w++){
        // (for (w, h) in this particular group, group is divided into num_PE_width groups)
        int thisGroupHeight = output_height / 2 + ((w / 2 == 0) ? output_height % 2 : 0);
        int thisGroupWidth = output_width / 2 + ((w % 2 == 0) ? output_width % 2 : 0);
        if (thisGroupHeight == 0 || thisGroupWidth == 0){
          continue;
        }

        int outputChannelGroup = outputStatus / (thisGroupHeight * thisGroupWidth); // TODO: we might have strange thing if we have different timing for the end of output channel
        int outputPositionIndex = outputStatus % (thisGroupHeight * thisGroupWidth);

        int writeOutputChannel = h + num_PE_height * outputChannelGroup;

        int writeOutputHeightPrefix = (w / 2 == 0) ? 0 : output_height - thisGroupHeight;
        int writeOutputHeight = writeOutputHeightPrefix + outputPositionIndex / thisGroupWidth;

        int writeOutputWidthPrefix = (w % 2 == 0) ? 0 : output_width - thisGroupWidth;
        int writeOutputWidth = writeOutputWidthPrefix + outputPositionIndex % thisGroupWidth;
        if (writeOutputChannel < num_output_channel && writeOutputHeight < output_height && writeOutputWidth < output_width){
          outputMemory[writeOutputChannel][writeOutputHeight][writeOutputWidth] = outputOfPEs[h][w];
        }
      }
    }
  };

  void writeOutput(
    std::vector<std::vector<int>>& outputOfPEs,
    std::vector<std::vector<int>>& outputExpOfPEs,
    std::vector<std::vector<std::vector<int>>>& outputMemory,
    std::vector<std::vector<std::vector<int>>>& outputExpMemory,
    int outputStatus, // index of output, when first output, it is zero
    int output_height,
    int output_width,
    int num_output_channel
  )
  {
    for (int h = 0; h < num_PE_height; h++)
    {
      for (int w = 0; w < num_PE_width; w++){
        // (for (w, h) in this particular group, group is divided into num_PE_width groups)
        int thisGroupHeight = output_height / 2 + ((w / 2 == 0) ? output_height % 2 : 0);
        int thisGroupWidth = output_width / 2 + ((w % 2 == 0) ? output_width % 2 : 0);
        if (thisGroupHeight == 0 || thisGroupWidth == 0){
          continue;
        }

        int outputChannelGroup = outputStatus / (thisGroupHeight * thisGroupWidth); // TODO: we might have strange thing if we have different timing for the end of output channel
        int outputPositionIndex = outputStatus % (thisGroupHeight * thisGroupWidth);

        int writeOutputChannel = h + num_PE_height * outputChannelGroup;

        int writeOutputHeightPrefix = (w / 2 == 0) ? 0 : output_height - thisGroupHeight;
        int writeOutputHeight = writeOutputHeightPrefix + outputPositionIndex / thisGroupWidth;

        int writeOutputWidthPrefix = (w % 2 == 0) ? 0 : output_width - thisGroupWidth;
        int writeOutputWidth = writeOutputWidthPrefix + outputPositionIndex % thisGroupWidth;
        if (writeOutputChannel < num_output_channel && writeOutputHeight < output_height && writeOutputWidth < output_width){
          outputMemory[writeOutputChannel][writeOutputHeight][writeOutputWidth] = outputOfPEs[h][w];
          outputExpMemory[writeOutputChannel][writeOutputHeight][writeOutputWidth] = outputExpOfPEs[h][w];
        }
      }
    }
  };

  void extractInputExpFromFifos(
    std::vector<std::vector<std::deque<int>>>& inputExpFifos,
    std::vector<DecodedRegister>& preDecodedInputs,
    std::vector<DecodedRegister>& decodedInputs,
    std::vector<int>& sharedExpForInputs,
    std::vector<int>& psumShiftedWidths
  )
  {
    for (int fifoIndex = 0; fifoIndex < num_PE_width; fifoIndex++){
      // get max for this block
      int maxExp = 0;
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        // if (input_channel == 0){std::cout << inputExpFifos[fifoIndex][input_channel].front() << std::endl;}
        maxExp = std::max(maxExp, inputExpFifos[fifoIndex][input_channel].front());
      }

      int nowExp = sharedExpForInputs[fifoIndex];
      int shiftedWidth = maxExp - nowExp;
      // std::cout << shiftedWidth << std::endl;

      sharedExpForInputs[fifoIndex] = maxExp;
      psumShiftedWidths[fifoIndex] = shiftedWidth;

      // shift the decodedInputs
      for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
        int shiftWidthForDecode = maxExp - inputExpFifos[fifoIndex][input_channel].front();
        // if (input_channel == 0) { std::cout << maxExp << " " << inputExpFifos[fifoIndex][input_channel].front() << " " << shiftWidthForDecode << std::endl;}
        // shift decodedInputs[fifoIndex][input_channel] by shiftWidthForDecode bits
        for (int bitIndex = 0; bitIndex < num_decodedRegister; bitIndex++){
          if (preDecodedInputs[fifoIndex].isValids[input_channel][bitIndex] && preDecodedInputs[fifoIndex].bitInputValues[input_channel][bitIndex] >= shiftWidthForDecode){
            decodedInputs[fifoIndex].bitInputValues[input_channel][bitIndex] = preDecodedInputs[fifoIndex].bitInputValues[input_channel][bitIndex] - shiftWidthForDecode;
            decodedInputs[fifoIndex].isNegatives[input_channel][bitIndex] = preDecodedInputs[fifoIndex].isNegatives[input_channel][bitIndex];
            decodedInputs[fifoIndex].isValids[input_channel][bitIndex] = preDecodedInputs[fifoIndex].isValids[input_channel][bitIndex];
            // std::cout << decodedInputs[fifoIndex].bitInputValues[input_channel][bitIndex] << std::endl;
          }
          else{
            decodedInputs[fifoIndex].bitInputValues[input_channel][bitIndex] = 0;
            decodedInputs[fifoIndex].isNegatives[input_channel][bitIndex] = false;
            decodedInputs[fifoIndex].isValids[input_channel][bitIndex] = false;
          }
          // std::cout << shiftWidthForDecode << " " << preDecodedInputs[fifoIndex].bitInputValues[input_channel][bitIndex] << " " << decodedInputs[fifoIndex].bitInputValues[input_channel][bitIndex] << std::endl;
        }
        // std::cout << shiftWidthForDecode << " " << std::endl;
        // std::cout << shiftWidthForDecode << " " << preDecodedInputs[fifoIndex].bitInputValues[input_channel] << " " << decodedInputs[fifoIndex].bitInputValues[input_channel] << std::endl;
      }
    }
  };

  void extractWeightExpFromFifos(
    std::vector<std::vector<std::deque<int>>>& weightExpFifos,
    std::vector<int>& sharedExpForWeights
  )
  {
    // weightExpFifos is already quantized so we do not need to shift anything
    for (int fifoIndex = 0; fifoIndex < num_PE_height; fifoIndex++){
      sharedExpForWeights[fifoIndex] = weightExpFifos[fifoIndex][0].front();
    }
  };
  #pragma endregion PEArrayUtils

  #pragma BitConversion
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

  float CreateFloatFromBFloat(std::pair<int,int> p){
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
  #pragma endregion BitConversion
}
