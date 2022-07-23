// 
//  task_2.c
//  https://www.tinkercad.com/things/kA52pKifDo5-workshop-task-2
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
const int BUTTON = 13;  //  Button Pin. Note that Pin 13 is a special one.

//  Store the individual states of the Red, Yellow, and Green 
//  lights (in that order) for the enumerated states listed above.
//  E.g. 0b100 represents Red is ON, Yellow and Green are OFF.
const int LED_STATE[] = {
  0b100,  //  STOP
  0b110,  //  READY
  0b001,  //  GO
  0b010,  //  SLOW
};

//  We'll declare a helper function which updates the LEDs based on the current state.
void update_leds();

void setup()
{
  //  Initialise LEDs.
  for (int i = 0; i < NUM_LEDS; ++i) 
    pinMode(LED[i], OUTPUT);
  
  //  Initialise Button.
  pinMode(BUTTON, INPUT);
  
  //  Show the initial state.
  update_leds();
}

void loop()
{
  //  We'll use this static variable to keep track of whether the 
  //  button was pressed on the previous loop.
  //  The `static` keyword means the variable and value are retained 
  //  across iterations.
  static bool prev_pressed = false;

  //  Read the state of the button. (1 / true = button is pressed)
  bool pressed = digitalRead(BUTTON);

  //  Detect the rising edge (it wasn't pressed before, 
  //  but it is now).
  if (!prev_pressed && pressed)
  {
    state = State((state + 1) % 4);
    update_leds();
  }
  
  //  Update our state variable.
  prev_pressed = pressed;
}

void update_leds()
{
  //  Write to each LED according to the current state.
  for (int led = 0; led < 3; ++led)
    digitalWrite(LED[led], LED_STATE[state] & (1 << led));
}
