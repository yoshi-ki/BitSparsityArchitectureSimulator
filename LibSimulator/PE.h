#pragma once
#include <vector>

namespace simulator
{
  class PE
  {
    public:
      /* initializer */
      PE();

      /* method */
      // execute one step and output psum
      // input: 3bit unsigned * 16 * 2
      // output: 22bit signed
      int execute_one_step(std::vector<unsigned int>& bitActivations, std::vector<unsigned int>& bitWeights);

      // for test and mock
      // output: 22bit signed
      int get_psum();


      /* member */
      int psum;
  };
}
