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
    auto a = std::vector<unsigned int>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    auto b = std::vector<unsigned int>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    pe.execute_one_step(a, b);

    // 2 step
    auto c = std::vector<unsigned int>{1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1};
    auto d = std::vector<unsigned int>{1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1};
    pe.execute_one_step(c, d);
    ASSERT_THAT(pe.get_psum(), Eq(44));
  }

  // test to check when the case of not-allowed inputs

  // test to output 22 bit constraint
}
