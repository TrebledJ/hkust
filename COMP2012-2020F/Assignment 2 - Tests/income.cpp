#include "income.h"


string to_string(GoesTo gt);


Income::Income(double amount, bool hasRealized, string date, TransactionCategory ic, GoesTo gt, string description)
    : Transaction{amount, hasRealized, std::move(date), std::move(description)}
    , sources{ic}
    , sumToWhere{gt}
{
    std::cout << "Income with amount " << amount << " dated " << this->date << " initialized." << std::endl;
}

Income::~Income()
{
    std::cout << "Income with " << amount << " dollars on " << date << " deleted." << std::endl;
}

void Income::setGoesTo(GoesTo gt)
{
    std::cout << "Income has sent to " << static_cast<int>(gt) << std::endl;
    sumToWhere = gt;
}

ostream& Income::printTransaction(ostream& out) const
{
    out << "=====================================" << std::endl;
    out << "Income: " << descriptions << std::endl;
    out << "Income Date: " << date << std::endl;
    out << "Amount: " << amount << std::endl;
    out << "Has been realized?: " << hasRealized << std::endl;
    out << "Amount added into: " << to_string(sumToWhere) << std::endl;
    return out << "=====================================" << std::endl;
}

TransactionCategory Income::getCategory() const
{
    std::cout << "Identifying which type of income is this..." << std::endl;
    return sources;
}

GoesTo Income::getGoesTo() const
{
    std::cout << "Identify where this income goes to..." << std::endl;
    return sumToWhere;
}

Income* Income::operator*(int salaryMultiplier)
{
    std::cout << "The transaction amount has been multiplied " << salaryMultiplier << " times." << std::endl;
    amount *= salaryMultiplier;
    return this;
}