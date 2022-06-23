#include "PEArray.h"
#include "PE.h"
#include "time.h"
#include <iostream>
#include <stdexcept> // std::runtime_error
#include "math.h"
#include <omp.h>


namespace simulator
{
  // bool that represents the PE Array is computing now or not

  // PE Array
  std::vector<std::vector<simulator::PE>> PEs(num_PE_height, std::vector<simulator::PE>(num_PE_width, PE()));

  // for unit test
  PEArray::PEArray(){};

  PEArray::PEArray(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
    int num_input_channel,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  )
  {
    busy = true;

    PEArray::outputStatus = 0;
    PEArray::input_height = input_height;
    PEArray::input_width = input_width;
    PEArray::kernel_height = kernel_height;
    PEArray::kernel_width = kernel_height;
    PEArray::stride = stride;
    PEArray::num_output_channel = num_output_channel;

    PEArray::output_height = ((input_height - kernel_height) / stride) + 1;
    PEArray::output_width = ((input_width - kernel_width) / stride) + 1;

    outputMemory = std::vector<std::vector<std::vector<int>>>(num_output_channel, std::vector<std::vector<int>>(output_height, std::vector<int>(output_width)));

    inputValuesFifos = std::vector<std::vector<std::deque<FIFOValues>>>(num_PE_width, std::vector<std::deque<FIFOValues>>(num_PE_parallel, std::deque<FIFOValues>()));
    weightValuesFifos = std::vector<std::vector<std::deque<FIFOValues>>> (num_PE_height, std::vector<std::deque<FIFOValues>>(num_PE_parallel, std::deque<FIFOValues>()));

    // convert input activation and weight to the bit format,
    // because we only need bit format value
    convertInputMemoriesToFifos(inputMemories, inputValuesFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);
    convertWeightMemoriesToFifos(weightMemories, weightValuesFifos, num_input_channel, input_height, input_width, kernel_height, kernel_width, stride, num_output_channel);

    // Initialize states to control PEs
    inputControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_width);
    weightControllerStatusForPEs = std::vector<PEControllerStatus>(num_PE_height);

    decodedInputs = v<DecodedRegister>(num_PE_width, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_PE_parallel)), v<v<bool>>(num_PE_parallel, v<bool>(8)), v<v<bool>>(num_PE_parallel, v<bool>(8))});
    decodedWeights = v<DecodedRegister>(num_PE_height, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_PE_parallel)), v<v<bool>>(num_PE_parallel, v<bool>(8)), v<v<bool>>(num_PE_parallel, v<bool>(8))});

    outputOfPEs = v<v<int>>(num_PE_height, std::vector<int>(num_PE_width));

    inputsForPEs = std::vector<PEInput>(num_PE_width, PEInput{v<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel)});
    weightsForPEs = std::vector<PEInput>(num_PE_height, PEInput{v<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel)});
  }

  bool PEArray::execute_one_step()
  {
    // std::cout << "input fifo count: " << inputValuesFifos[0][0].size() << "first value: " << (inputValuesFifos[0][0].size() != 0 ? inputValuesFifos[0][0].front().value : 0)<< std::endl;
    // std::cout << "weight fifo count: " << weightValuesFifos[0][0].size() << "weight value: " << (weightValuesFifos[0][0].size() != 0 ? weightValuesFifos[0][0].front().value : 0)<< std::endl;
    // if all of the value fifos become empty, we finish execution
    if(isLayerFinished(inputValuesFifos, weightValuesFifos)){
      // output for debug
      // std::cout << "finished layer execution" << std::endl;
      busy = false;
      return busy;
    }

    // clock_t start1 = clock();

    // Initialize bit representations vector. 8 is for the max bit size of the value
    // decodedInputs = v<DecodedRegister>(num_PE_width, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_PE_parallel)), v<v<bool>>(num_PE_parallel, v<bool>(8)), v<v<bool>>(num_PE_parallel, v<bool>(8))});
    // decodedWeights = v<DecodedRegister>(num_PE_height, DecodedRegister{v<v<unsigned int>>(num_PE_parallel, v<unsigned>(num_PE_parallel)), v<v<bool>>(num_PE_parallel, v<bool>(8)), v<v<bool>>(num_PE_parallel, v<bool>(8))});

    // decode values (this circuit always run) (just look at the first element and do not change fifo)
    decodeValuesToBits(inputValuesFifos, decodedInputs);
    decodeValuesToBits(weightValuesFifos, decodedWeights);
    // clock_t start2 = clock();

    // based on the PE controller status, prepare input for PEs.
    // inputsForPEs and weightsForPEs are represented by bit representation.
    // auto inputsForPEs = std::vector<PEInput>(num_PE_width, PEInput{v<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel)});
    // auto weightsForPEs = std::vector<PEInput>(num_PE_height, PEInput{v<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel)});
    // we check the pe status and decide we send values or not and which to send for PEs.
    createInputForPEsBasedOnControllerStatus(decodedInputs, inputControllerStatusForPEs, inputsForPEs, num_PE_width);
    createInputForPEsBasedOnControllerStatus(decodedWeights, weightControllerStatusForPEs, weightsForPEs, num_PE_height);

    // clock_t start3 = clock();
    // we use
    // std::vector<std::vector<int>> outputOfPEs(num_PE_height, std::vector<int>(num_PE_width));
    #pragma omp parallel num_threads(4)
    for (int h = 0; h < num_PE_height; h++)
    {
      for (int w = 0; w < num_PE_width; w++){
        outputOfPEs[h][w] = PEs[h][w].execute_one_step(inputsForPEs[w], weightsForPEs[h]);
        // std::cout << h << " " << w << " " << outputOfPEs[h][w] << std::endl;
        // if (h == 2 && w == 2){
        //   std::cout << "input" << inputValuesFifos[h][w].front().value << std::endl;
        //   std::cout << "input: " << inputsForPEs[w].bitInputValue[0] << "isNegative: " << inputsForPEs[w].isNegative[0] << "isValid: " << inputsForPEs[w].isValid[0] << std::endl;
        //   std::cout << "weight" << weightValuesFifos[h][w].front().value << std::endl;
        //   std::cout << "weight: " << weightsForPEs[h].bitInputValue[0] << "isNegative:" << weightsForPEs[h].isNegative[0] << "isValid: " << weightsForPEs[h].isValid[0] << std::endl;
        //   std::cout << h << " " << w << " " << outputOfPEs[h][w] << std::endl;
        // }
      }
    }

    // clock_t start4 = clock();

    // update pe status
    updatePEStatus(inputControllerStatusForPEs, weightControllerStatusForPEs, inputValuesFifos, weightValuesFifos, decodedInputs, decodedWeights);

    // clock_t end = clock();

    // std::cout << static_cast<double>(start2 - start1) << " " << static_cast<double>(start3 - start2) << " " << static_cast<double>(start4 - start3) << " " << static_cast<double>(end - start4)  << std::endl;

    bool finishedPsumExecution = isFinishedPSumExecution(inputControllerStatusForPEs, weightControllerStatusForPEs);

    // for mock of step 1, we write output if all PEs finish
    if (finishedPsumExecution)
    {
      // TODO: something wired might happen when 0 input
      // write the output of PEs to the corresponding output position
      writeOutput(outputOfPEs, outputMemory, outputStatus, output_height, output_width, num_output_channel);
      outputStatus = outputStatus + 1;

      // update PE Status again to read next layers
      // finishedPSum -> false, isWaiting -> false, index -> 0, pop fifo
      updatePEStatusWhenPsumFinish(inputControllerStatusForPEs, weightControllerStatusForPEs, inputValuesFifos, weightValuesFifos);

      // reset state inside PE
      for (int h = 0; h < num_PE_height; h++){
        for (int w = 0; w < num_PE_width; w++){
          PEs[h][w].reset_state();
        }
      }
    }
    return busy;
  };

  bool PEArray::isLayerFinished(
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

  bool PEArray::isFinishedPSumExecution(
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

  void PEArray::decodeValuesToBits(
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

  void PEArray::createInputForPEsBasedOnControllerStatus(
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

  void PEArray::updatePEStatus(
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
        // std::cout << decodedInputs[fifoIndex].isValids[bitIndex][nextProcessIndex] << std::endl;

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

  void PEArray::updatePEStatusWhenPsumFinish(
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

  void PEArray::convertInputMemoriesToFifos(
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
        // int partialInputHeight = (memoryIndex == 0 || memoryIndex == 1) ? input_height / 2 + input_height % 2 : input_height / 2;
        // int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? input_width / 2 + input_width % 2 : input_width / 2;

        int num_output_height = ((input_height - kernel_height) / stride) + 1;
        int num_output_width = ((input_width - kernel_width) / stride) + 1;
        int firstGroupOutputHeight = num_output_height / 2 + num_output_height % 2;
        int firstGroupOutputWidth = num_output_width / 2 + num_output_width % 2;
        int partialInputHeight = (memoryIndex == 0 || memoryIndex == 1) ? firstGroupOutputHeight : num_output_height - firstGroupOutputHeight;
        int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? firstGroupOutputWidth : num_output_width - firstGroupOutputWidth;

        // std::cout << partialInputHeight << partialInputWidth << std::endl;
        // start position of window
        for (int windowStartHeight = 0; windowStartHeight < partialInputHeight; windowStartHeight = windowStartHeight + stride){
          for (int windowStartWidth = 0; windowStartWidth < partialInputWidth; windowStartWidth = windowStartWidth + stride){
            for (int kh = 0; kh < kernel_height; kh++){
              for (int kw = 0; kw < kernel_width; kw++){
                for (int channelGroup = 0; channelGroup < num_input_channel_group; channelGroup++){
                  // input
                  for (int input_channel = 0; input_channel < num_PE_parallel; input_channel++){
                    inputValuesFifos[memoryIndex][input_channel].push_back(
                        FIFOValues{
                            inputMemories[memoryIndex][windowStartHeight + kh][windowStartWidth + kw][channelGroup][input_channel],
                            false});
                    // if (input_channel == 0){
                    //   std::cout << inputValuesFifos[memoryIndex][input_channel].end()->value << std::endl;
                    // }
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

  void PEArray::convertWeightMemoriesToFifos(
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
      // int partialInputHeight = (memoryIndex == 0 || memoryIndex == 1) ? input_height / 2 + input_height % 2 : input_height / 2;
      // int partialInputWidth = (memoryIndex == 0 || memoryIndex == 2) ? input_width / 2 + input_width % 2 : input_width / 2;

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

  void PEArray::writeOutput(
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
        // std::cout << outputStatus << " " << thisGroupHeight << " " << thisGroupWidth << std::endl;
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
        // std::cout << "outputMemoryPlace: " << writeOutputChannel << " " << writeOutputHeight << " " << writeOutputWidth << std::endl;
        if (writeOutputChannel < num_output_channel && writeOutputHeight < output_height && writeOutputWidth < output_width){
          outputMemory[writeOutputChannel][writeOutputHeight][writeOutputWidth] = outputOfPEs[h][w];
          // std::cout << outputOfPEs[h][w] << std::endl;
          // std::cout << outputMemory[writeOutputChannel][writeOutputHeight][writeOutputWidth] << std::endl;
        }
      }
    }
  };

}