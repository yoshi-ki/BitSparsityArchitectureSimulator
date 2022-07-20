#pragma once
#include <vector>
#include <deque>

namespace simulator
{
  const int num_PE_height = 8;
  const int num_PE_width = 4;
  const int num_PE_parallel = 16; // PE consumes 16 bits at once
  const int num_bit_size = 8;
  const int num_decodedRegister = 8;

  struct PEControllerStatus{
    std::vector<bool> isWaiting = std::vector<bool>(num_PE_parallel);
    std::vector<int> nextProcessIndex = std::vector<int>(num_PE_parallel);
    std::vector<bool> finishedPSum = std::vector<bool>(num_PE_parallel);
  };

  struct FIFOValues{
    int value;
    bool isLast;
  };

  struct BFloatFIFOValues{
    int value;
    int expValue;
    bool isLast;
  };

  struct DecodedRegister{
    // it is for 16 inputs and values just decoded
    std::vector<std::vector<unsigned int>> bitInputValues; // [inputChannel][bitIndex]
    std::vector<std::vector<bool>> isNegatives;
    std::vector<std::vector<bool>> isValids;
  };

  struct PEInput{
    // it is for 16 inputs
    std::vector<unsigned int> bitInputValue;
    std::vector<bool> isNegative;
    std::vector<bool> isValid;
  };


  void convertInputToInputMemoryFormat(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>> &inputMemories,
    int stride,
    int num_kernel_height,
    int num_kernel_width
  );

  void convertWeightToWeightMemoryFormat(
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>> &weightMemories
  );

  void computeConv(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<int>>> &outputValues,
    int stride
  );

  void computeConvFloat(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<int>>> &inputExpValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightExpValues,
    std::vector<std::vector<std::vector<float>>> &outputValues,
    int stride
  );


  // PE rray Utils functions

  void convertInputMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
    int num_input_channel_group,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  );

  void convertInputMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>& inputExpMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& inputValuesFifos,
    std::vector<std::vector<std::deque<int>>>& inputExpFifos,
    int num_input_channel_group,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  );


  void convertWeightMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
    int num_input_channel_group,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  );

  void convertWeightMemoriesToFifos(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightMemories,
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<int>>>>>>& weightExpMemories,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightValuesFifos,
    std::vector<std::vector<std::deque<int>>>& weightExpFifos,
    int num_input_channel_group,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride,
    int num_output_channel
  );

  bool isLayerFinished(
    std::vector<std::vector<std::deque<FIFOValues>>>& inputFifos,
    std::vector<std::vector<std::deque<FIFOValues>>>& weightFifos
  );

  bool isFinishedPSumExecution(
    std::vector<PEControllerStatus>& inputControllerStatusForPEs,
    std::vector<PEControllerStatus>& weightControllerStatusForPEs
  );

  void decodeValuesToBits(
    std::vector<std::vector<std::deque<FIFOValues>>> &valueFifos,
    std::vector<DecodedRegister>& decodedRepresentations
  );

  void decodeValuesToBitsWithLeadingOne(
    std::vector<std::vector<std::deque<FIFOValues>>> &valueFifos,
    std::vector<std::vector<std::deque<int>>> &expFifos,
    std::vector<DecodedRegister>& decodedRepresentations
  );

  void createInputForPEsBasedOnControllerStatus(
    std::vector<DecodedRegister>& decodedRegisters,
    std::vector<PEControllerStatus>& controllerStatusForPEs,
    std::vector<PEInput>& representationsForPEs,
    int num_Fifo
  );

  void updatePEStatus(
    std::vector<PEControllerStatus> &inputControllerStatusForPEs,
    std::vector<PEControllerStatus> &weightControllerStatusForPEs,
    std::vector<std::vector<std::deque<FIFOValues>>> &inputValuesFifos,
    std::vector<std::vector<std::deque<FIFOValues>>> &weightValuesFifos,
    std::vector<DecodedRegister> &decodedInputs,
    std::vector<DecodedRegister> &decodedWeights
  );

  void updatePEStatusWhenPsumFinish(
    std::vector<PEControllerStatus> & inputControllerStatusForPEs,
    std::vector<PEControllerStatus> & weightControllerStatusForPEs,
    std::vector<std::vector<std::deque<FIFOValues>>> & inputValuesFifos,
    std::vector<std::vector<std::deque<FIFOValues>>> & weightValuesFifos
  );

  void writeOutput(
    std::vector<std::vector<int>>& outputOfPEs,
    std::vector<std::vector<std::vector<int>>>& outputMemory,
    int outputStatus,
    int output_height,
    int output_width,
    int num_output_channel
  );

  void writeOutput(
    std::vector<std::vector<int>>& outputOfPEs,
    std::vector<std::vector<int>>& outputExpOfPEs,
    std::vector<std::vector<std::vector<int>>>& outputMemory,
    std::vector<std::vector<std::vector<int>>>& outputExpMemory,
    int outputStatus,
    int output_height,
    int output_width,
    int num_output_channel
  );

  void extractInputExpFromFifos(
    std::vector<std::vector<std::deque<int>>>& inputExpFifos,
    std::vector<DecodedRegister>& preDecodedInputs,
    std::vector<DecodedRegister>& decodedInputs,
    std::vector<int>& sharedExpForInputs,
    std::vector<int>& psumShiftedWidths
  );

  void extractWeightExpFromFifos(
    std::vector<std::vector<std::deque<int>>>& weightExpFifos,
    std::vector<int>& sharedExpForWeights
  );






  // Bit Conversion
  std::pair<int,int> CreateBFloatFromFloat(float v);
  float CreateFloatFromBFloat(std::pair<int, int> p);
}
