#include "BFloatPE.h"
#include <iostream>
#include <stdexcept> // std::runtime_error
#include <omp.h>
#include <bitset>

namespace simulator
{
  union int_uint{
        unsigned ui;
        int i;
    } int_uint;

  BFloatPE::BFloatPE(){
    psum = 0;
  }

  int BFloatPE::execute_one_step(
    PEInput bitActivations,
    PEInput bitWeights,
    int psumShiftedWidth
  )
  {
    if (psum != 0){
      for (size_t i = 0; i < bitActivations.bitInputValue.size(); ++i) {
        std::cout << bitActivations.bitInputValue.at(i) << " " <<  bitActivations.isValid.at(i) << " " << bitActivations.isNegative.at(i) << "; ";
      }
      std::cout << std::endl;
      for (size_t i = 0; i < bitWeights.bitInputValue.size(); ++i) {
        std::cout << bitWeights.bitInputValue.at(i) << " " <<  bitWeights.isValid.at(i) << " " << bitWeights.isNegative.at(i) << "; ";
      }
      std::cout << std::endl;
    }

    // if (psumShiftedWidth != 0){
    //   std::cout << psum << std::endl;
    // }
    psum = psum >> psumShiftedWidth;
    // if (psumShiftedWidth != 0){
    //   std::cout << psum << " " << psumShiftedWidth<< std::endl;
    // }
    // compute 16 multiplications
    // #pragma omp parallel for
    for (int i = 0; i < num_PE_parallel; i++){
      if (bitActivations.isValid[i] && bitWeights.isValid[i]){
        // TODO: change this to 22 bit integer
        auto multIsNegative = bitActivations.isNegative[i] ^ bitWeights.isNegative[i];
        auto multExp = bitActivations.bitInputValue[i] + bitWeights.bitInputValue[i];
        // #pragma omp atomic
        psum += multIsNegative ? (-(1 << multExp)) : (1 << multExp);
      }
    }
    if (psum != 0){
      std::cout <<  std::bitset<25>(psum) << std::endl;
    }
    return psum;
  }

  void BFloatPE::reset_state(){
    psum = 0;
  }

  int FindFirstOnePosition(unsigned x) {
    int ct=0;
    while (x > 1) { ct++; x = x >> 1; }
    return ct;
  }

  std::pair<int,int> BFloatPE::get_psum(int inputExp, int weightExp)
  {
    // output bfloat values
    // std::cout << inputExp << " " << weightExp << " " << psum << std::endl;

    // std::cout << std::bitset<25>(psum) << std::endl;
    // convert to correct float format
    if (psum == 0){
      return std::pair<int,int>{0, psum};
    }
    int abspsum = abs(psum);
    int_uint.i = abspsum;
    unsigned u_psum = int_uint.ui;
    int firstOnePosition = FindFirstOnePosition(u_psum);
    int exp = inputExp + weightExp - 127 + firstOnePosition - 14;
    u_psum = u_psum - (1 << firstOnePosition);
    // std::cout << firstOnePosition << " " << std::bitset<32>(u_psum) << std::endl;
    if (firstOnePosition >= 7){
      u_psum = u_psum >> (firstOnePosition - 7);
    }
    else{
      u_psum = u_psum << (-firstOnePosition + 7);
    }

    // add sign bit
    if(psum < 0){
      u_psum += 128;
    }

    // std::cout << exp << " " << u_psum << std::endl;

    int_uint.ui = u_psum;
    // std::cout << int_uint.i << std::endl;
    return std::pair<int, int>{std::max(exp,0), int_uint.i};
  }
}
