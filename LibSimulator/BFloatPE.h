#pragma once
#include <vector>
#include "PEArray.h"

namespace simulator
{

  class BFloatPE
  {
    public:
      /* initializer */
      BFloatPE();

      /* method */
      // execute one step and output psum
      // input: 3bit unsigned * 16 * 2
      // output: 22bit signed
      int execute_one_step(
        PEInput bitActivations,
        PEInput bitWeights
      );

      // output: 22bit signed
      int get_psum(int actExp, int weightExp);

      // reset psum and other state for next iteration of PE
      void reset_state();

      /* member */
      int psum;
  };
}
