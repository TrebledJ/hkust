// 
//  task_f1_lo.c (lights only)
//  https://www.tinkercad.com/things/lBKUoZ2XJcx-workshop-f1-task
// 

#include <time.h>

//  Define a set of states which allow us to 
//  conveniently write stuff like `state = BLANK;`.
enum State {
  COUNTDOWN,
  FORMATION,
  BLANK,
  GO,

  NUM_STATES,
} prev_state, state = COUNTDOWN; //  The current state, initially set.
//  Try changing the initial state from `state = COUNTDOWN` to `state = FORMATION;`
//  To see a different sequence of lights.

//  Define some constants.
const int NUM_LEDS = 5;
const int RED_LED[] = {4, 3, 2, 1, 0}; // Red LED Pins, ordered from LEFT to RIGHT.
const int GREEN_LED[] = {10, 9, 8, 7, 6}; //  Green LED Pins, ordered from LEFT to RIGHT.

const int COUNTDOWN_DELAY = 1000;
const int FORMATION_DELAY = 2000;
int counter = 0;  //  Use a variable to keep track of counting.

/// @brief  Checks if counting has finished. If so, performs some state-specific action.
void check_counting_finished();

/// @brief  Checks if the state has changed.
bool state_changed();

/// @brief  Checks if the LED needs to be updated.
bool led_update_needed();

/**
 * @brief Returns the LED pattern in the 10 rightmost bits.
 *        0b G4 ... G0 R4 ... R0 where R0, G0 are the left LEDs.
 */
int get_led_pattern(State state, int counter);

void setup()
{
  //  Seed randomness.
  srand(time(0));

  //  Initialise the LEDs.
  for (int i = 0; i < NUM_LEDS; ++i) 
  {
    pinMode(RED_LED[i], OUTPUT);
    pinMode(GREEN_LED[i], OUTPUT);
  }

  prev_state = state;
}

void loop()
{
  check_counting_finished();
  
  if (led_update_needed())
  {
    if (state_changed())
    {
      //  Reset the counter if state has changed.
      counter = 0;
    }

    //  Get the pattern for this state and counter.
    int ptn = get_led_pattern(state, counter);
    
    //  Update LEDs.
    for (int i = 0; i < NUM_LEDS; ++i)
    {
      digitalWrite(RED_LED[i], ptn & (1 << i));
      digitalWrite(GREEN_LED[i], bool(ptn & (1 << (NUM_LEDS + i))));
    }

    //  Update counter and delay.
    switch (state)
    {
    case COUNTDOWN:
      counter++;
      delay(COUNTDOWN_DELAY);
      break;
    case FORMATION:
      counter++;
      delay(FORMATION_DELAY);
      break;
    default:
      break;
    }
  }

  prev_state = state;
}

void check_counting_finished()
{
  //  Check if counting has finished.
  switch (state)
  {
  case COUNTDOWN:
    if (counter == NUM_LEDS + 1)
    {
      delay(COUNTDOWN_DELAY + rand() % 4000); //  Delay an additional 0-4 seconds.
      state = BLANK;
    }
    break;

  case FORMATION:
    if (counter == NUM_LEDS)
      state = GO;
    break;

  default: break;
  }
}

bool state_changed()
{
  return prev_state != state;
}

bool led_update_needed()
{
  return state_changed() || state == COUNTDOWN || state == FORMATION;
}

int get_led_pattern(State state, int counter)
{
  static const int cd_ptns[NUM_LEDS + 1] = {0b00000, 0b00001, 0b00011, 0b00111, 0b01111, 0b11111};
  static const int fmt_ptns[NUM_LEDS + 1] = {0b11111, 0b11110, 0b11100, 0b11000, 0b10000, 0b00000};

  switch (state)
  {
  case COUNTDOWN: return cd_ptns[counter];
  case FORMATION: return fmt_ptns[counter];
  case BLANK: return 0b0;
  case GO: return 0b1111100000;
  default: return 0;
  }
}