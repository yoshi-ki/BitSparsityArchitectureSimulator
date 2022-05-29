#include "gmock/gmock.h"
#include <iostream>
#include <gtest/internal/gtest-port.h>
#include "PE.h"

using namespace testing;

namespace simulator::tests
{
	TEST(ExampleTests, Example) {
		simulator::PE pe = simulator::PE();
        ASSERT_THAT(pe.get_psum(), Eq(0));
	}

  // test to check if we can compute several steps
  TEST(PETest, NormalAddition) {
		simulator::PE pe = simulator::PE();
    // 1 step

    auto a = PEInput{std::vector<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel, true)};
    auto b = PEInput{std::vector<unsigned int>(num_PE_parallel), v<bool>(num_PE_parallel), v<bool>(num_PE_parallel, true)};
    pe.execute_one_step(a, b);
    ASSERT_THAT(pe.get_psum(), Eq(16));

    // 2 step
    auto c = PEInput{std::vector<unsigned int>{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, v<bool>(num_PE_parallel), v<bool>(num_PE_parallel, true)};
    auto d = PEInput{std::vector<unsigned int>{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, v<bool>(num_PE_parallel), v<bool>(num_PE_parallel, true)};
    d.isNegative[0] = true;
    c.isNegative[1] = true;
    c.isValid[2] = false;
    d.isValid[2] = false;
    pe.execute_one_step(c, d);
    ASSERT_THAT(pe.get_psum(), Eq(24));
  }

  // test to check when the case of not-allowed inputs

  // test to output 22 bit constraint
}
