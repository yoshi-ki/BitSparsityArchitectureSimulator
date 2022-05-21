#pragma once
#include <vector>

// OverAllArch calls PEArray once for each layer
// this will be the interface with main function for simulator

namespace simulator
{

  struct outputOverAllArch{
    // output vector
  };
  class OverAllArch
  {
    public:
      /* initializer */
      OverAllArch();

      /* method */
      // execute one step and output vector
      void execute();

      // output: vector
      // void send_output();

      // input: vector
      // void get_input();

      // reset psum and other state for next iteration of PE
      void reset_state();

      /* member */
  };
}
