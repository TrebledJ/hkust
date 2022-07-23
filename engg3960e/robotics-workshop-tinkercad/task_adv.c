// 
//  task_adv.c
//  https://www.tinkercad.com/things/cngvMtkXLG8-workshop-adv-task
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
const int LED[] = {2, 4, 7}; //  LED Pins, ordered from Green to Red.
const int LED_PWM[] = {11, 10, 3};  // LED PWM Pins, ordered as RBG.
const int BUTTON = 13;  //  Button Pin. Note that Pin 13 is a special one.

//	Array of GO durations.
const int GO_TIMES[] = {1000, 2000, 5000, 10000, 20000};
const int NUM_GO_TIMES = sizeof(GO_TIMES) / sizeof(int);

//  Store the individual states of the Red, Yellow, and Green 
//  lights (in that order) for the enumerated states listed above.
//  E.g. 0b100 represents Red is ON, Yellow and Green are OFF.
const int LED_STATE[] = {
  0b100,  //  STOP
  0b110,  //  READY
  0b001,  //  GO
  0b010,  //  SLOW
};

//  Enum for the RYG LEDs.
enum LEDColor { RED = 2, YELLOW = 1, GREEN = 0 };

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
  //  The `static` keyword means the variable and value are retained 
  //  across iterations.
  static int go_index = 0; // Index of which GO duration to use.
  static int next_change = 0; // Keeps track of when the state should next change.
  
  //  We'll use this static variable to keep track of whether the 
  //  button was pressed on the previous loop.
  static bool prev_pressed = false;
  
  //  The next block of code could've been replaced by a `delay(time);`
  //  But the problem with this is that button presses would not be
  //  recorded DURING THE DELAY... so it'd be extremely difficult to 
  //  get the button press to register.
  
  // Check if the state should change.
  int now = millis();
  if (now > next_change)
  {
    // Do le change!
    state = State((state + 1) % 4);
  	update_leds();
    
    // Record the next time the state should change.
    int time = state == GO ? GO_TIMES[go_index] : 1000;
    next_change = now + time;
  }

  //  Read the state of the button. (1 / true = button is pressed)
  bool pressed = digitalRead(BUTTON);

  //  Detect the rising edge (it wasn't pressed before, 
  //  but it is now).
  if (!prev_pressed && pressed)
  {
    go_index = (go_index + 1) % NUM_GO_TIMES;
  }
  
  //  Update our state variable.
  prev_pressed = pressed;
  
  //  We'll add a small delay because we need to be kind to the simulator.
  delay(10);
}

void update_leds()
{
  //  Update RYG LEDs.
  //  Write to each LED according to the current state.
  for (int led = 0; led < 3; ++led)
    digitalWrite(LED[led], LED_STATE[state] & (1 << led));
  
  //  Update RGB LED.
  int red = state == READY ? 255 : (255 * (LED_STATE[state] & (1 << RED)) + 64 * (LED_STATE[state] & (1 << YELLOW))) % 256;
  int blue = 0;
  int green = state == READY ? 140 : (255 * (LED_STATE[state] & (1 << GREEN)) + 64 * (LED_STATE[state] & (1 << YELLOW))) % 256;
  analogWrite(LED_PWM[0], red);
  analogWrite(LED_PWM[1], blue);
  analogWrite(LED_PWM[2], green);
}
