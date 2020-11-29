#include "expenses.h"


string to_string(GoesTo gt);


Expenses::Expenses(double amount, bool hasRealized, string date, TransactionCategory ic, GoesTo gt, string description)
    : Transaction{amount, hasRealized, std::move(date), std::move(description)}
    , sources{ic}
    , sumToWhere{gt}
{
    std::cout << "Expense with amount " << amount << " dated " << this->date << " initialized." << std::endl;
}

Expenses::~Expenses() 
{
    std::cout << "Expense with " << amount << " dollars on " << date << " deleted." << std::endl;
}

void Expenses::setGoesTo(GoesTo gt)
{
    std::cout << "Expense has sent to " << static_cast<int>(gt) << std::endl;
    sumToWhere = gt;
}
TransactionCategory Expenses::getCategory() const
{
    std::cout << "Identifying which type of expenses is this..." << std::endl;
    return sources;
}
GoesTo Expenses::getGoesTo() const
{
    std::cout << "Identify where this expense goes to..." << std::endl;
    return sumToWhere;
}

ostream& Expenses::printTransaction(ostream& out) const
{ 
    out << "=====================================" << std::endl;
    out << "Expense: " << descriptions << std::endl;
    out << "Expense Date: " << date << std::endl;
    out << "Amount: " << amount << std::endl;
    out << "Has been realized?: " << hasRealized  << std::endl;
    out << "Amount added into: " << to_string(sumToWhere) << std::endl;;
    return out << "=====================================" << std::endl;
}
