#include "util.h"

void Util::printMatrix()
{
    MatrixXd m(2,2);
    m(0,0) = 1;
    m(1,0) = 2;
    m(0,1) = 3;
    m(1,1) = 4;
    std::cout << "Here is the matrix m:\n" << m << std::endl;
   
  
}