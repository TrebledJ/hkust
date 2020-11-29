#include "transactions.h"


string to_string(GoesTo gt)
{
    const char* s[] = {"Bank", "Cash", "Credit Card"};
    return s[gt];
}


Transaction::Transaction(double amount, bool hasRealized, string date, string describe)
    : amount{amount}
    , hasRealized{hasRealized}
    , date{std::move(date)}
    , descriptions{std::move(describe)}
{
}

Transaction::~Transaction()
{
    std::cout << "Transaction with " << amount << " dollars on " << date << " deleted." << std::endl;
}

void Transaction::setAmount(double amount)
{
    std::cout << "Transaction with " << amount << " dollars on " << date << " added." << std::endl;
    this->amount = amount;
}
double Transaction::getAmount() const { return amount; }

void Transaction::setHasRealized()
{
    std::cout << "The transaction has realized already!" << std::endl;
    hasRealized = true;
}
bool Transaction::getHasRealized() { return hasRealized; }

void Transaction::setDate(string str) { date = std::move(str); }
string Transaction::getDate() const { return date; }

void Transaction::setDescriptions(string str) { descriptions = std::move(str); }
string Transaction::getDescriptions() const { return descriptions; }

ostream& operator<< (ostream &out, const Transaction &trans)
{
    return trans.printTransaction(out);
}
