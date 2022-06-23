#include <cstdlib>
#include <iostream>

#include "PE.h"

int main(int argc, char** argv)
{
    simulator::PE pe = simulator::PE();
    std::cout << "hello world => " << pe.get_psum() << std::endl;

    return EXIT_SUCCESS;
}
