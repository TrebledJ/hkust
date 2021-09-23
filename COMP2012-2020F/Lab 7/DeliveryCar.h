#ifndef DELIVERYCAR_H
#define DELIVERYCAR_H

#include "Vehicle.h"


class DeliveryCar : protected Vehicle
{
public:
    DeliveryCar(int nc, Color c, int m, int charge);
    ~DeliveryCar();

    int getChargePerMile() const;
    void setChargePerMile(int charge);

    void start();
    void brake(int);

    virtual void print() const override;

private:
    int m_chargePerMile;
};


#endif
