#pragma once
#include <vector>

namespace simulator
{
  struct outputPE{
    int psum;
    bool endComputation;
  };

  class PE
  {
    public:
      /* initializer */
      PE();

      /* method */
      // execute one step and output psum
      // input: 3bit unsigned * 16 * 2
      // output: 22bit signed
      outputPE execute_one_step(
        std::vector<unsigned int>& bitActivations,
        std::vector<unsigned int>& bitWeights,
        bool activationEnd,
        bool weightEnd
      );

      // for test and mock
      // output: 22bit signed
      int get_psum();

      // reset psum and other state for next iteration of PE
      void reset_state();

      /* member */
      int psum;
      int state;
      bool endComputation;
  };
}
