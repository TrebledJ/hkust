#ifndef PA_UTILS_H
#define PA_UTILS_H

#include <iostream>
#include <string>


/// @brief  Displays a header marking the beginning of a problem.
void problem_header(const std::string& name)
{
    const std::string eqs((CONSOLE_WIDTH - 2 - 7 - 5 - name.size()) / 2, '=');
    std::cout << "\n[" << eqs + (name.size() % 2 ? "=" : "") << " Problem <" << name << "> " << eqs << "]" << std::endl;
}


#endif