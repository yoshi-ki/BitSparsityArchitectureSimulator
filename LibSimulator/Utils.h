#include "PE.h"
#include "PEArray.h"

namespace simulator
{
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

}
