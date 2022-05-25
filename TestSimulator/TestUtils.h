namespace simulator::tests
{
  void makeRandomInput(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>& inputMemories,
    std::vector<std::vector<std::vector<int>>>& inputValues,
    int num_input_channel,
    int num_input_height,
    int num_input_width,
    std::set<int>& availableValueSet
  );

  void makeRandomWeight(
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<std::int8_t>>>>>> &weightMemories,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    int num_kernel_height,
    int num_kernel_width,
    int num_input_channel,
    int num_output_channel,
    std::set<int> & availableValueSet
  );

  void computeConv(
    std::vector<std::vector<std::vector<int>>> &inputValues,
    std::vector<std::vector<std::vector<std::vector<int>>>> &weightValues,
    std::vector<std::vector<std::vector<int>>> &outputValues,
    int stride
  );
}