// 
//  task_1.c
//  https://www.tinkercad.com/things/8EtHA1svgu9-workshop-task-1
// 

//  Define a set of traffic light states which allow us to 
//  conveniently write stuff like `state = GO;`.
enum State {
  STOP,
  READY,
  GO,
  SLOW,
} state = STOP; //  The current state, initially set to STOP.

//  Define some constants.
const int NUM_STATES = 4;
const int NUM_LEDS = 3;
const int LED[] = {2, 4, 7}; // LED Pins, ordered from Green to Red.
const int BUTTON = 12;  //  Button Pin.

//  Store the individual states of the Red, Yellow, and Green 
//  lights (in that order) for the enumerated states listed above.
//  E.g. 0b100 represents Red is ON, Yellow and Green are OFF.
const int LED_STATE[] = {
  0b100,  //  STOP
  0b110,  //  READY
  0b001,  //  GO
  0b010,  //  SLOW
};

void setup()
{
  //  Initialise the LEDs.
  for (int i = 0; i < NUM_LEDS; ++i) 
    pinMode(LED[i], OUTPUT);
}

void loop()
{
  delay(1000); // Wait for 1000 millisecond(s)
  
  //  Write to each LED according to the current state.
  for (int led = 0; led < 3; ++led)
    digitalWrite(LED[led], LED_STATE[state] & (1 << led));
  
  //  Increment the state and take its remainder (%) so that it 
  //  will loop from 0 to NUM_STATES over and over again.
  state = State((state + 1) % NUM_STATES);
}